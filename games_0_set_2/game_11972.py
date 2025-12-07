import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:48:58.918831
# Source Brief: brief_01972.md
# Brief Index: 1972
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Bio-Evolve: DNA Lab - A Gymnasium Environment

    In this puzzle game, the agent controls a cursor to manipulate a DNA sequence.
    The goal is to match a target sequence within a time limit.

    - Harvest energy particles by clicking on them.
    - Use energy to craft and apply enzymes.
    - Enzymes mutate the DNA in specific ways.
    - A time-slowing mechanic (Shift) helps in planning moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manipulate a DNA sequence to match a target. Harvest energy particles to power enzymes that "
        "mutate the DNA, and slow down time to plan your moves."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to harvest energy or apply the "
        "selected enzyme. Hold shift to slow down time."
    )
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    MAX_TIME = 1200 # 40 seconds at 30 FPS
    
    # Colors
    COLOR_BG = (16, 16, 32)
    COLOR_GRID = (32, 32, 64)
    COLOR_DNA_BACKBONE = (64, 128, 128)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_GLOW = (255, 255, 0, 50)
    COLOR_ENERGY = (0, 255, 0)
    COLOR_ENERGY_GLOW = (0, 255, 0, 50)
    
    BASE_COLORS = {
        'A': (80, 220, 80),   # Green
        'T': (220, 80, 80),   # Red
        'C': (80, 80, 220),   # Blue
        'G': (220, 220, 80)   # Yellow
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0.0
        self.energy = 0.0
        self.cursor_pos = pygame.math.Vector2(0, 0)
        self.current_dna = []
        self.target_dna = []
        self.dna_level = 1
        self.particles = []
        self.effects = []
        self.enzymes = {}
        self.selected_enzyme_key = None
        self.last_space_held = False
        self.last_num_matches = 0
        self.reward_this_step = 0.0
        
        # self.reset() is called by the wrapper, no need to call it here
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_TIME
        self.energy = 25.0
        self.cursor_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        # --- DNA Sequence Generation ---
        dna_length = 2 + self.dna_level * 2
        bases = list(self.BASE_COLORS.keys())
        self.target_dna = [self.np_random.choice(bases) for _ in range(dna_length)]
        self.current_dna = list(self.target_dna)
        # Ensure current_dna is different from target
        while self.current_dna == self.target_dna:
            idx_to_change = self.np_random.integers(0, dna_length)
            possible_bases = [b for b in bases if b != self.target_dna[idx_to_change]]
            self.current_dna[idx_to_change] = self.np_random.choice(possible_bases)

        self.last_num_matches = self._count_matches()
        
        # --- Particles & Effects ---
        self.particles = []
        for _ in range(5):
            self._spawn_particle()
        self.effects = []
        
        # --- Enzymes ---
        self.enzymes = {
            'SWAP_AT': {'cost': 10, 'name': 'SWAP A/T', 'unlocked': True},
            'SWAP_CG': {'cost': 10, 'name': 'SWAP C/G', 'unlocked': True},
            'CYCLE_ALL': {'cost': 15, 'name': 'CYCLE', 'unlocked': True},
        }
        self.selected_enzyme_key = 'SWAP_AT'
        
        self.last_space_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.reward_this_step = 0.0
        self.steps += 1
        
        # --- Action Unpacking ---
        movement, space_now_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_now_held and not self.last_space_held
        self.last_space_held = space_now_held
        
        # --- Game Logic Update ---
        self._handle_input(movement, space_pressed)
        self._update_game_state(shift_held)

        # --- Reward Calculation ---
        # Base reward for surviving
        self.reward_this_step -= 0.01 
        
        # Reward for improving DNA match
        current_matches = self._count_matches()
        if current_matches > self.last_num_matches:
            self.reward_this_step += (current_matches - self.last_num_matches) * 0.5
        self.last_num_matches = current_matches
        
        self.score += self.reward_this_step
        
        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = False # This env doesn't truncate
        if terminated:
            self.game_over = True
            if self.current_dna == self.target_dna:
                self.reward_this_step += 100.0 # Victory
                self.score += 100.0
                self.dna_level += 1 # Difficulty progression for next game
            else:
                self.reward_this_step -= 100.0 # Timeout
                self.score -= 100.0

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )
    
    # --- Helper Methods for step() ---
    
    def _handle_input(self, movement, space_pressed):
        # Cursor Movement
        cursor_speed = 5
        if movement == 1: self.cursor_pos.y -= cursor_speed
        elif movement == 2: self.cursor_pos.y += cursor_speed
        elif movement == 3: self.cursor_pos.x -= cursor_speed
        elif movement == 4: self.cursor_pos.x += cursor_speed
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

        if space_pressed:
            self._handle_click()

    def _handle_click(self):
        # 1. Check for particle harvesting
        for p in self.particles[:]:
            if self.cursor_pos.distance_to(p['pos']) < p['radius']:
                # SFX: energy_harvest.wav
                self.energy += p['value']
                self.reward_this_step += p['value'] * 0.1
                self._add_effect('text', pos=p['pos'], text=f"+{p['value']:.0f}", color=self.COLOR_ENERGY)
                self.particles.remove(p)
                self._spawn_particle()
                return

        # 2. Check for DNA mutation
        dna_rects = self._get_dna_rects()
        for i, rect in enumerate(dna_rects):
            if rect.collidepoint(self.cursor_pos):
                self._apply_enzyme(i)
                return

        # 3. Check for enzyme selection
        enzyme_rects = self._get_enzyme_rects()
        for key, rect in enzyme_rects.items():
            if rect.collidepoint(self.cursor_pos):
                # SFX: ui_click.wav
                self.selected_enzyme_key = key
                return

    def _apply_enzyme(self, dna_index):
        if not self.selected_enzyme_key: return
        
        enzyme = self.enzymes[self.selected_enzyme_key]
        if self.energy >= enzyme['cost']:
            # SFX: dna_mutate.wav
            self.energy -= enzyme['cost']
            self.reward_this_step += 1.0 # Small reward for taking a meaningful action
            
            base = self.current_dna[dna_index]
            new_base = base
            
            if self.selected_enzyme_key == 'SWAP_AT':
                if base == 'A': new_base = 'T'
                elif base == 'T': new_base = 'A'
            elif self.selected_enzyme_key == 'SWAP_CG':
                if base == 'C': new_base = 'G'
                elif base == 'G': new_base = 'C'
            elif self.selected_enzyme_key == 'CYCLE_ALL':
                cycle = ['A', 'T', 'C', 'G', 'A']
                if base in cycle:
                    new_base = cycle[cycle.index(base) + 1]

            self.current_dna[dna_index] = new_base
            
            dna_rect = self._get_dna_rects()[dna_index]
            self._add_effect('pulse', pos=dna_rect.center, color=self.BASE_COLORS[new_base])
        else:
            # SFX: error.wav
            self._add_effect('text', pos=self.cursor_pos, text="NO ENERGY", color=(255, 50, 50))


    def _update_game_state(self, shift_held):
        time_dilation = 0.25 if shift_held else 1.0
        self.timer -= 1.0 * time_dilation
        
        # Update particles
        for p in self.particles:
            p['pos'] += p['vel'] * time_dilation
            if p['pos'].x < 0 or p['pos'].x > self.SCREEN_WIDTH: p['vel'].x *= -1
            if p['pos'].y < 0 or p['pos'].y > self.SCREEN_HEIGHT: p['vel'].y *= -1
        
        # Update effects
        for e in self.effects[:]:
            e['life'] -= 1
            if e['type'] == 'text':
                e['pos'].y -= 0.5
            if e['life'] <= 0:
                self.effects.remove(e)

    def _check_termination(self):
        win = self.current_dna == self.target_dna
        timeout = self.timer <= 0
        max_steps = self.steps >= self.MAX_STEPS
        return win or timeout or max_steps

    def _count_matches(self):
        return sum(1 for i, j in zip(self.current_dna, self.target_dna) if i == j)

    # --- Spawning Methods ---
    
    def _spawn_particle(self):
        pos = pygame.math.Vector2(
            self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
            self.np_random.uniform(50, self.SCREEN_HEIGHT - 100)
        )
        vel = pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
        if vel.length() > 0:
            vel.normalize_ip()
        self.particles.append({
            'pos': pos, 'vel': vel, 'radius': 10, 'value': 10.0
        })

    def _add_effect(self, type, pos, **kwargs):
        effect = {'type': type, 'pos': pygame.math.Vector2(pos), 'life': 30}
        effect.update(kwargs)
        self.effects.append(effect)
        
    # --- Rendering Methods ---
    
    def _get_observation(self):
        # This is the main render loop
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_dna()
        self._render_effects()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            # Glow effect
            for i in range(p['radius'], 0, -2):
                alpha = int(self.COLOR_ENERGY_GLOW[3] * (i / p['radius']))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, (*self.COLOR_ENERGY_GLOW[:3], alpha))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius'] * 0.7), self.COLOR_ENERGY)

    def _render_dna(self):
        # Target DNA
        self._draw_text("TARGET", (self.SCREEN_WIDTH / 2, 30), self.font_medium, self.COLOR_TEXT)
        self._render_dna_strand(self.target_dna, self.SCREEN_HEIGHT * 0.25, is_target=True)
        
        # Current DNA
        self._draw_text("CURRENT", (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 10), self.font_medium, self.COLOR_TEXT)
        self._render_dna_strand(self.current_dna, self.SCREEN_HEIGHT * 0.5, is_target=False)

    def _render_dna_strand(self, dna_strand, y_center, is_target):
        num_bases = len(dna_strand)
        base_size = 30
        spacing = 15
        total_width = num_bases * base_size + (num_bases - 1) * spacing
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        
        # Draw backbone
        backbone_y1 = y_center - base_size / 2 - 5
        backbone_y2 = y_center + base_size / 2 + 5
        pygame.draw.line(self.screen, self.COLOR_DNA_BACKBONE, (start_x, backbone_y1), (start_x + total_width, backbone_y1), 3)
        pygame.draw.line(self.screen, self.COLOR_DNA_BACKBONE, (start_x, backbone_y2), (start_x + total_width, backbone_y2), 3)

        for i, base in enumerate(dna_strand):
            x = start_x + i * (base_size + spacing)
            rect = pygame.Rect(x, y_center - base_size / 2, base_size, base_size)
            
            # Draw connector
            pygame.draw.line(self.screen, self.COLOR_DNA_BACKBONE, (rect.centerx, backbone_y1), (rect.centerx, backbone_y2), 3)

            # Draw base
            color = self.BASE_COLORS[base]
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if not is_target:
                is_correct = base == self.target_dna[i]
                border_color = (255, 255, 255) if is_correct else (80, 80, 80)
                pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=4)
            
            self._draw_text(base, rect.center, self.font_medium, (0,0,0))
    
    def _render_cursor(self):
        pos = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        radius = 12
        # Glow
        for i in range(radius, 0, -2):
            alpha = int(self.COLOR_CURSOR_GLOW[3] * (i / radius)**2)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, (*self.COLOR_CURSOR_GLOW[:3], alpha))
        # Core
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius*0.5), self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius*0.5)-1, self.COLOR_CURSOR)
        
    def _render_effects(self):
        for e in self.effects:
            if e['type'] == 'text':
                alpha = int(255 * (e['life'] / 30))
                color = (*e['color'], alpha)
                self._draw_text(e['text'], e['pos'], self.font_small, color, use_alpha=True)
            elif e['type'] == 'pulse':
                radius = int(30 * (1 - e['life'] / 30))
                alpha = int(100 * (e['life'] / 30))
                color = (*e['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(e['pos'][0]), int(e['pos'][1]), radius, color)

    def _render_ui(self):
        # Bottom UI Panel
        ui_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 80, self.SCREEN_WIDTH, 80)
        s = pygame.Surface((self.SCREEN_WIDTH, 80), pygame.SRCALPHA)
        s.fill((0,0,0,150))
        self.screen.blit(s, (0, self.SCREEN_HEIGHT - 80))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.SCREEN_HEIGHT - 80), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 80))

        # Timer Bar
        timer_percent = max(0, self.timer / self.MAX_TIME)
        timer_width = (self.SCREEN_WIDTH - 20) * timer_percent
        timer_color = (255, 255, 0) if timer_percent > 0.5 else (255, 128, 0) if timer_percent > 0.2 else (255, 0, 0)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, 10, self.SCREEN_WIDTH - 20, 10))
        pygame.draw.rect(self.screen, timer_color, (10, 10, timer_width, 10))
        
        # Score
        self._draw_text(f"SCORE: {self.score:.2f}", (10, self.SCREEN_HEIGHT - 25), self.font_small, self.COLOR_TEXT, align='left')

        # Energy
        self._draw_text(f"ENERGY: {self.energy:.0f}", (self.SCREEN_WIDTH / 4, self.SCREEN_HEIGHT - 60), self.font_medium, self.COLOR_ENERGY)

        # Enzymes
        self._draw_text("ENZYMES", (self.SCREEN_WIDTH * 0.65, self.SCREEN_HEIGHT - 60), self.font_medium, self.COLOR_TEXT)
        enzyme_rects = self._get_enzyme_rects()
        for key, rect in enzyme_rects.items():
            enzyme = self.enzymes[key]
            is_selected = self.selected_enzyme_key == key
            can_afford = self.energy >= enzyme['cost']
            
            color = (50, 50, 80)
            if is_selected:
                color = (80, 80, 120)
            
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 2, border_radius=4)
            
            text_color = self.COLOR_TEXT if can_afford else (120, 120, 120)
            self._draw_text(f"{enzyme['name']} ({enzyme['cost']})", rect.center, self.font_small, text_color)

    # --- Geometry and Text Helpers ---
    
    def _get_dna_rects(self):
        rects = []
        dna_strand = self.current_dna
        y_center = self.SCREEN_HEIGHT * 0.5
        num_bases = len(dna_strand)
        base_size = 30
        spacing = 15
        total_width = num_bases * base_size + (num_bases - 1) * spacing
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        for i in range(num_bases):
            x = start_x + i * (base_size + spacing)
            rects.append(pygame.Rect(x, y_center - base_size / 2, base_size, base_size))
        return rects

    def _get_enzyme_rects(self):
        rects = {}
        start_x = self.SCREEN_WIDTH * 0.65 - 60
        y_pos = self.SCREEN_HEIGHT - 35
        width = 120
        height = 25
        
        enzyme_keys = list(self.enzymes.keys())
        total_width = len(enzyme_keys) * width + (len(enzyme_keys) - 1) * 10
        start_x = self.SCREEN_WIDTH * 0.75 - total_width / 2

        for i, key in enumerate(enzyme_keys):
            rects[key] = pygame.Rect(start_x + i * (width + 10), y_pos, width, height)
        return rects

    def _draw_text(self, text, pos, font, color, align='center', use_alpha=False):
        if use_alpha:
            text_surface = font.render(text, True, color[:3])
            text_surface.set_alpha(color[3])
        else:
            # Shadow effect for better readability
            shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surface.get_rect()
            if align == 'center': shadow_rect.center = (pos[0] + 1, pos[1] + 1)
            elif align == 'left': shadow_rect.topleft = (pos[0] + 1, pos[1] + 1)
            self.screen.blit(shadow_surface, shadow_rect)
            
            text_surface = font.render(text, True, color)

        text_rect = text_surface.get_rect()
        if align == 'center': text_rect.center = pos
        elif align == 'left': text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "timer": self.timer,
            "dna_level": self.dna_level,
            "matches": self.last_num_matches,
            "target_length": len(self.target_dna)
        }
        
    def close(self):
        pygame.quit()
        

# --- Example Usage ---
if __name__ == '__main__':
    # This block is for manual play and will not be run by the evaluation server.
    # It is used for debugging and testing the environment.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display
    env = GameEnv()
    
    pygame.display.set_caption("Bio-Evolve: DNA Lab")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print("--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")
    
    while not done:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_q:
                    done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()