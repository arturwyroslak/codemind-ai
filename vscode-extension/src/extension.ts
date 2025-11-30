import * as vscode from 'vscode';
import axios, { AxiosError } from 'axios';

let diagnosticCollection: vscode.DiagnosticCollection;
let outputChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
    console.log('CodeMind AI is now active!');

    outputChannel = vscode.window.createOutputChannel('CodeMind AI');
    diagnosticCollection = vscode.languages.createDiagnosticCollection('codemind');
    
    context.subscriptions.push(diagnosticCollection);
    context.subscriptions.push(outputChannel);

    // Command: Analyze selection
    const analyzeSelection = vscode.commands.registerCommand('codemind.analyzeSelection', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);

        if (!selectedText) {
            vscode.window.showInformationMessage('Please select some code to analyze');
            return;
        }

        await analyzeCode(selectedText, editor, selection);
    });

    // Command: Analyze current file
    const analyzeFile = vscode.commands.registerCommand('codemind.analyzeFile', async (uri?: vscode.Uri) => {
        let document: vscode.TextDocument;
        
        if (uri) {
            document = await vscode.workspace.openTextDocument(uri);
        } else {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No file to analyze');
                return;
            }
            document = editor.document;
        }

        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const fullText = document.getText();
        const fullRange = new vscode.Range(0, 0, document.lineCount, 0);
        
        await analyzeCode(fullText, editor, fullRange);
    });

    // Command: Apply suggested fix
    const applyFix = vscode.commands.registerCommand('codemind.applyFix', async (diagnostic: vscode.Diagnostic) => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || !diagnostic) {
            return;
        }

        // Extract fix from diagnostic message
        const fixMatch = diagnostic.message.match(/\[FIX\]\s*(.*)/s);
        if (!fixMatch) {
            vscode.window.showInformationMessage('No fix suggestion available');
            return;
        }

        const fixCode = fixMatch[1].trim();
        
        await editor.edit(editBuilder => {
            editBuilder.replace(diagnostic.range, fixCode);
        });

        vscode.window.showInformationMessage('✅ CodeMind AI fix applied!');
    });

    // Auto-analyze on save
    const onSave = vscode.workspace.onDidSaveTextDocument(async (document) => {
        const config = vscode.workspace.getConfiguration('codemind');
        if (config.get('autoAnalyzeOnSave') as boolean) {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document === document) {
                const fullText = document.getText();
                const fullRange = new vscode.Range(0, 0, document.lineCount, 0);
                await analyzeCode(fullText, editor, fullRange);
            }
        }
    });

    context.subscriptions.push(analyzeSelection, analyzeFile, applyFix, onSave);
}

async function analyzeCode(
    code: string,
    editor: vscode.TextEditor,
    range: vscode.Range | vscode.Selection
) {
    const config = vscode.workspace.getConfiguration('codemind');
    const apiUrl = config.get('apiUrl') as string;
    const apiToken = config.get('apiToken') as string;
    const analysisType = config.get('analysisType') as string;
    const enableInline = config.get('enableInlineDiagnostics') as boolean;

    if (!apiToken) {
        vscode.window.showErrorMessage('CodeMind AI: Please set your API token in settings');
        return;
    }

    outputChannel.appendLine(`[${new Date().toISOString()}] Starting analysis...`);
    outputChannel.show(true);

    // Show progress
    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'CodeMind AI',
        cancellable: false
    }, async (progress) => {
        progress.report({ message: 'Analyzing code...' });

        try {
            const response = await axios.post(
                `${apiUrl}/api/v1/advanced/analyze`,
                {
                    code: code,
                    language: editor.document.languageId,
                    analysis_type: analysisType,
                    context: {
                        file_path: editor.document.fileName,
                        selection_start: range.start.line,
                        selection_end: range.end.line
                    }
                },
                {
                    headers: {
                        'Authorization': `Bearer ${apiToken}`,
                        'Content-Type': 'application/json'
                    },
                    timeout: 30000  // 30 seconds timeout
                }
            );

            const result = response.data;
            displayResults(result, editor, enableInline);
            
            vscode.window.showInformationMessage(
                `✅ Analysis complete! Score: ${result.overall_score?.toFixed(1) || 'N/A'}/100`
            );

        } catch (error) {
            handleError(error as AxiosError);
        }
    });
}

function displayResults(
    result: any,
    editor: vscode.TextEditor,
    enableInline: boolean
) {
    outputChannel.clear();
    outputChannel.appendLine('='.repeat(60));
    outputChannel.appendLine('CodeMind AI Analysis Results');
    outputChannel.appendLine('='.repeat(60));
    outputChannel.appendLine(`Overall Score: ${result.overall_score?.toFixed(1) || 'N/A'}/100`);
    outputChannel.appendLine(`Analysis Type: ${result.analysis_type || 'full'}`);
    outputChannel.appendLine(`Execution Time: ${result.execution_time?.toFixed(2) || 'N/A'}s`);
    outputChannel.appendLine('');

    const findings = result.findings || [];

    if (findings.length === 0) {
        outputChannel.appendLine('✨ No issues found! Your code looks great!');
        diagnosticCollection.clear();
        return;
    }

    outputChannel.appendLine(`Found ${findings.length} issue(s):\n`);

    // Display in output channel
    findings.forEach((finding: any, index: number) => {
        const icon = getSeverityIcon(finding.severity);
        outputChannel.appendLine(`${index + 1}. ${icon} [Line ${finding.line}] ${finding.message}`);
        outputChannel.appendLine(`   Severity: ${finding.severity}`);
        outputChannel.appendLine(`   Type: ${finding.type}`);
        if (finding.fix) {
            outputChannel.appendLine(`   💡 Suggested Fix: ${finding.fix}`);
        }
        outputChannel.appendLine('');
    });

    // Display as inline diagnostics
    if (enableInline) {
        const diagnostics: vscode.Diagnostic[] = [];

        findings.forEach((finding: any) => {
            const line = Math.max(0, (finding.line || 1) - 1);
            const lineText = editor.document.lineAt(line).text;
            const range = new vscode.Range(
                line, 0,
                line, lineText.length
            );

            let message = `${finding.message}`;
            if (finding.fix) {
                message += `\n💡 Suggested Fix: ${finding.fix}`;
                message += `\n[FIX] ${finding.fix}`;
            }

            const diagnostic = new vscode.Diagnostic(
                range,
                message,
                mapSeverity(finding.severity)
            );

            diagnostic.code = 'codemind-ai';
            diagnostic.source = 'CodeMind AI';
            diagnostics.push(diagnostic);
        });

        diagnosticCollection.set(editor.document.uri, diagnostics);
    }
}

function mapSeverity(severity: string): vscode.DiagnosticSeverity {
    switch (severity?.toLowerCase()) {
        case 'critical':
            return vscode.DiagnosticSeverity.Error;
        case 'high':
            return vscode.DiagnosticSeverity.Warning;
        case 'medium':
            return vscode.DiagnosticSeverity.Information;
        case 'low':
            return vscode.DiagnosticSeverity.Hint;
        default:
            return vscode.DiagnosticSeverity.Information;
    }
}

function getSeverityIcon(severity: string): string {
    switch (severity?.toLowerCase()) {
        case 'critical': return '🔴';
        case 'high': return '🟠';
        case 'medium': return '🟡';
        case 'low': return '🟢';
        default: return 'ℹ️';
    }
}

function handleError(error: AxiosError) {
    let errorMessage = 'Unknown error occurred';

    if (error.response) {
        const status = error.response.status;
        const detail = (error.response.data as any)?.detail || error.message;

        if (status === 401) {
            errorMessage = '🔒 Authentication failed. Please check your API token.';
        } else if (status === 429) {
            errorMessage = '⏱️ Rate limit exceeded. Please try again later.';
        } else if (status >= 500) {
            errorMessage = `🔥 Server error: ${detail}`;
        } else {
            errorMessage = `❌ Error (${status}): ${detail}`;
        }
    } else if (error.request) {
        errorMessage = '🌐 Network error: Could not reach CodeMind AI server. Check your API URL.';
    } else {
        errorMessage = `❌ ${error.message}`;
    }

    outputChannel.appendLine(`\n[ERROR] ${errorMessage}`);
    vscode.window.showErrorMessage(`CodeMind AI: ${errorMessage}`);
}

export function deactivate() {
    if (diagnosticCollection) {
        diagnosticCollection.dispose();
    }
    if (outputChannel) {
        outputChannel.dispose();
    }
    console.log('CodeMind AI deactivated');
}