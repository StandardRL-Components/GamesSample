import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:28:03.604625
# Source Brief: brief_03415.md
# Brief Index: 3415
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment simulating RNA sequencing and cell cycle management.

    The player manipulates an RNA sequence and the flow of time to match a target
    sequence, triggering a successful gene expression and cell cycle progression.
    Incorrect sequences damage the cell, eventually leading to its death. The goal
    is to complete 5 cell cycles before the cell dies or time runs out.

    Visuals are stylized and abstract, prioritizing clarity and "game feel" with
    smooth animations, particle effects, and clear UI feedback.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Simulate RNA sequencing by matching a target sequence to advance the cell cycle. "
        "Manage time and cell health to complete all cycles before the cell dies."
    )
    user_guide = (
        "Use ←→ to modify your RNA sequence and ↑↓ to control the speed of time. "
        "Match the target sequence to progress."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CYCLES_TO_WIN = 5
    MAX_STEPS = 2000
    BASE_HEALTH_DECAY = 0.05
    EXPRESSION_DAMAGE = 34.0
    NUCLEOTIDES = ['A', 'C', 'G', 'U']

    # --- Colors ---
    COLOR_BG = (10, 15, 25)
    COLOR_MEMBRANE = (40, 80, 120)
    COLOR_MEMBRANE_GLOW = (60, 120, 180)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_DIM = (150, 150, 170)
    COLOR_SUCCESS = (100, 255, 150)
    COLOR_ERROR = (255, 100, 100)
    COLOR_TIME_FX = (100, 180, 255)
    NUCLEOTIDE_COLORS = {
        'A': (255, 120, 120),
        'C': (120, 255, 120),
        'G': (120, 120, 255),
        'U': (255, 255, 120)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_ui = pygame.font.SysFont("Consolas", 16)
        self.font_feedback = pygame.font.SysFont("Verdana", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("Verdana", 48, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.cycles_completed = 0
        self.cell_health = 100.0
        self.time_multiplier = 1.0
        self.player_sequence = []
        self.target_sequence = []
        self.gene_expression_progress = 0.0
        self.particles = []
        self.feedback_messages = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.cycles_completed = 0
        self.cell_health = 100.0
        self.time_multiplier = 1.0
        self.player_sequence = []
        self.target_sequence = self._generate_target_sequence()
        self.gene_expression_progress = 0.0
        self.particles = []
        self.feedback_messages = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Action ---
        self._handle_action(action)

        # --- 2. Update Game State ---
        self.gene_expression_progress += self.time_multiplier * 0.5
        self.cell_health = max(0, self.cell_health - self.BASE_HEALTH_DECAY)
        self._update_particles()
        self._update_feedback_messages()

        # --- 3. Check for Gene Expression Event ---
        if self.gene_expression_progress >= 100:
            # Event: Gene expression completes
            self.gene_expression_progress = 0
            if self.player_sequence == self.target_sequence:
                # --- SUCCESS ---
                # sfx: success_chime.wav
                self.cycles_completed += 1
                self.target_sequence = self._generate_target_sequence()
                self.player_sequence = []
                self._add_feedback("SEQUENCE MATCH", self.COLOR_SUCCESS)
                self._spawn_particles(50, self.COLOR_SUCCESS, 2.0)
                reward += 10
            else:
                # --- FAILURE ---
                # sfx: error_buzz.wav
                self.cell_health = max(0, self.cell_health - self.EXPRESSION_DAMAGE)
                self._add_feedback("SEQUENCE MISMATCH", self.COLOR_ERROR)
                self._spawn_particles(30, self.COLOR_ERROR, 1.0)
                
        # --- 4. Calculate Reward & Check Termination ---
        reward += 1 # Survival reward
        
        terminated = False
        if self.cell_health <= 0:
            terminated = True
            reward = -100
            self._add_feedback("CELL DEATH", self.COLOR_ERROR, is_permanent=True)
        elif self.cycles_completed >= self.CYCLES_TO_WIN:
            terminated = True
            reward += 100
            self._add_feedback("CYCLE COMPLETE!", self.COLOR_SUCCESS, is_permanent=True)
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            reward = -50 # Penalize for running out of time

        if terminated or truncated:
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Action 0: No-op
        # Action 1 (Up): Increase time multiplier
        if movement == 1:
            self.time_multiplier = min(5.0, self.time_multiplier + 0.1)
            # sfx: time_up.wav
        # Action 2 (Down): Decrease time multiplier
        elif movement == 2:
            self.time_multiplier = max(0.1, self.time_multiplier - 0.1)
            # sfx: time_down.wav
        # Action 3 (Left): Remove last nucleotide
        elif movement == 3:
            if self.player_sequence:
                self.player_sequence.pop()
                # sfx: remove_nucleotide.wav
        # Action 4 (Right): Add a random nucleotide
        elif movement == 4:
            if len(self.player_sequence) < 15: # Limit sequence length
                new_nucleotide = self.np_random.choice(self.NUCLEOTIDES)
                self.player_sequence.append(new_nucleotide)
                # sfx: add_nucleotide.wav

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cell_health": self.cell_health,
            "cycles_completed": self.cycles_completed
        }

    def _render_game(self):
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        
        # Render time flow effect
        self._render_time_effect(center_x, center_y)

        # Render cell membrane with pulsating glow
        base_radius = 120
        pulsation = math.sin(self.steps * 0.05) * 5
        radius = int(base_radius + pulsation)
        
        # Glow effect
        for i in range(15, 0, -2):
            glow_alpha = 30 - i * 2
            color = (*self.COLOR_MEMBRANE_GLOW, glow_alpha)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius + i, color)

        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_MEMBRANE)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_MEMBRANE)
        
        # Render gene expression progress
        if self.gene_expression_progress > 0:
            progress_radius = int(radius * (self.gene_expression_progress / 100.0))
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, progress_radius, self.COLOR_MEMBRANE_GLOW)

        # Render particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['size']), color)

    def _render_time_effect(self, cx, cy):
        num_lines = 12
        angle_offset = (self.steps * self.time_multiplier * 0.01) % (2 * math.pi)
        for i in range(num_lines):
            angle = (i / num_lines) * 2 * math.pi + angle_offset
            length_mult = 1.0 + 0.1 * math.sin(self.steps * 0.05 + i)
            start_radius = 140 * length_mult
            end_radius = 160 * length_mult
            
            start_pos = (cx + math.cos(angle) * start_radius, cy + math.sin(angle) * start_radius)
            end_pos = (cx + math.cos(angle) * end_radius, cy + math.sin(angle) * end_radius)
            
            alpha = 50 + 20 * (self.time_multiplier - 1.0)
            alpha = int(max(0, min(255, alpha)))
            
            pygame.draw.line(self.screen, (*self.COLOR_TIME_FX, alpha), start_pos, end_pos, 2)


    def _render_ui(self):
        # --- Top UI: Score and Cycle ---
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))
        
        cycle_text = self.font_ui.render(f"CYCLE: {self.cycles_completed + 1}/{self.CYCLES_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(cycle_text, (self.SCREEN_WIDTH - cycle_text.get_width() - 15, 10))

        # --- Bottom UI: Health and Time ---
        # Health Bar
        health_perc = self.cell_health / 100.0
        health_color = self.COLOR_SUCCESS if health_perc > 0.6 else (255, 255, 0) if health_perc > 0.3 else self.COLOR_ERROR
        bar_width = 200
        bar_height = 15
        pygame.draw.rect(self.screen, (50,50,50), (15, self.SCREEN_HEIGHT - 25, bar_width, bar_height))
        pygame.draw.rect(self.screen, health_color, (15, self.SCREEN_HEIGHT - 25, int(bar_width * health_perc), bar_height))
        health_label = self.font_ui.render("CELL HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_label, (15, self.SCREEN_HEIGHT - 45))

        # Time Multiplier
        time_text = self.font_ui.render(f"TIME FLOW: x{self.time_multiplier:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 15, self.SCREEN_HEIGHT - 28))

        # --- Sequences ---
        self._render_sequence("TARGET", self.target_sequence, 15, 50)
        self._render_sequence("INPUT", self.player_sequence, 15, 100)
        
        # --- Feedback Messages ---
        y_offset = self.SCREEN_HEIGHT // 2 - 50
        for msg in self.feedback_messages:
            text_surf = self.font_feedback.render(msg['text'], True, msg['color'])
            text_surf.set_alpha(msg['alpha'])
            pos = (self.SCREEN_WIDTH // 2 - text_surf.get_width() // 2, y_offset)
            self.screen.blit(text_surf, pos)
            y_offset += 30

        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.cycles_completed >= self.CYCLES_TO_WIN else "GAME OVER"
            color = self.COLOR_SUCCESS if self.cycles_completed >= self.CYCLES_TO_WIN else self.COLOR_ERROR
            
            end_text = self.font_gameover.render(msg, True, color)
            pos = (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2)
            self.screen.blit(end_text, pos)

    def _render_sequence(self, label, sequence, x, y):
        label_text = self.font_main.render(f"{label}:", True, self.COLOR_TEXT_DIM)
        self.screen.blit(label_text, (x, y))
        
        box_size = 24
        padding = 4
        start_x = x + label_text.get_width() + 10
        
        for i, nucleotide in enumerate(sequence):
            px = start_x + i * (box_size + padding)
            color = self.NUCLEOTIDE_COLORS.get(nucleotide, (100, 100, 100))
            pygame.draw.rect(self.screen, color, (px, y, box_size, box_size), border_radius=4)
            
            char_text = self.font_main.render(nucleotide, True, (0,0,0))
            text_pos = (px + box_size // 2 - char_text.get_width() // 2, y + box_size // 2 - char_text.get_height() // 2)
            self.screen.blit(char_text, text_pos)
            
    def _generate_target_sequence(self):
        length = 2 + self.cycles_completed
        return [self.np_random.choice(self.NUCLEOTIDES) for _ in range(length)]

    def _spawn_particles(self, count, color, speed_mult):
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, 3.0) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(30, 61)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.98  # friction
            p['vel'][1] *= 0.98
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _add_feedback(self, text, color, duration=90, is_permanent=False):
        self.feedback_messages.append({
            'text': text,
            'color': color,
            'life': float('inf') if is_permanent else duration,
            'max_life': float('inf') if is_permanent else duration,
            'alpha': 255
        })

    def _update_feedback_messages(self):
        for msg in self.feedback_messages:
            msg['life'] -= 1
            if msg['max_life'] != float('inf'):
                # Fade out effect
                msg['alpha'] = int(255 * (msg['life'] / msg['max_life']))
        self.feedback_messages = [m for m in self.feedback_messages if m['life'] > 0]

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a display, so it will not run in a truly headless environment
    # To run, comment out the `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` line
    
    # Create a new GameEnv instance with the default 'rgb_array' render mode
    # but we will render it to a pygame display window
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup pygame display
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("GeneCycle Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # The game logic doesn't currently use space or shift, but we can still send the actions
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}. Press 'R' to restart.")

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()