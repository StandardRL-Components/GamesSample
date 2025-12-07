import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:16:45.023600
# Source Brief: brief_01597.md
# Brief Index: 1597
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Cosmic Rhapsody: A rhythm/simulation game where you build a spaceship
    and survive cosmic storms by playing rhythmic sequences.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Build your spaceship and survive cosmic storms by hitting rhythmic sequences to power your shields."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to navigate menus and space to select. "
        "Use ←↑→↓ arrow keys to hit notes during a storm. Hold shift in the build phase to clone a pilot."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors (Synthwave Palette)
    COLOR_BG = (13, 10, 33)
    COLOR_NEBULA_1 = (46, 15, 87, 50)
    COLOR_NEBULA_2 = (99, 24, 138, 40)
    COLOR_PLAYER = (0, 255, 255) # Cyan
    COLOR_PLAYER_GLOW = (0, 255, 255, 100)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_ACCENT = (255, 0, 191) # Magenta
    COLOR_SUCCESS = (0, 255, 128) # Bright Green
    COLOR_FAIL = (255, 50, 50) # Bright Red
    COLOR_NOTE_UP = (255, 230, 0) # Yellow
    COLOR_NOTE_DOWN = (0, 128, 255) # Blue
    COLOR_NOTE_LEFT = (255, 0, 191) # Magenta
    COLOR_NOTE_RIGHT = (255, 107, 0) # Orange
    NOTE_COLORS = [COLOR_NOTE_UP, COLOR_NOTE_DOWN, COLOR_NOTE_LEFT, COLOR_NOTE_RIGHT]

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State (persistent across resets) ---
        self.high_score = 0

        # --- Initialize episode-specific state ---
        # self.reset() is called by the wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Episode State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = 'build' # 'build' or 'storm'

        # Resources & Ship
        self.ship_hp = 100
        self.max_ship_hp = 100
        self.ship_parts = 0
        self.pilots = 1
        self.last_damage_time = -100 # Step counter for damage effect

        # Progression
        self.successful_storms = 0
        self.storm_level = 3 # Number of notes in sequence

        # Build Phase State
        self.build_menu_index = 0
        self.build_options = [
            {"label": "Build Shield Matrix", "cost": 25, "resource": "pilots"},
            {"label": "Engage Cosmic Storm", "cost": 0, "resource": None}
        ]

        # Storm Phase State
        self.storm_sequence = []
        self.storm_progress_index = 0
        self.note_y_pos = [] # Y position for each note in the sequence
        self.note_hit_feedback = [] # 0: pending, 1: hit, -1: miss
        self.player_input_history = [] # For visual feedback
        self.note_speed = 4.0

        # Visual Effects
        self.particles = []
        self.stars = [(self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.integers(1, 3)) for _ in range(150)]
        self.nebula_particles = [(self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.integers(40, 81)) for _ in range(15)]
        self.screen_shake = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle player action based on game phase ---
        if not self.game_over:
            if self.game_phase == 'build':
                reward += self._handle_build_phase(movement, space_press, shift_press)
            elif self.game_phase == 'storm':
                reward += self._handle_storm_phase(movement)

        # --- Update Game Systems ---
        self._update_particles()
        if self.screen_shake > 0: self.screen_shake -= 1
        self.steps += 1

        # --- Check for Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if (terminated or truncated) and not self.game_over:
            self.game_over = True
            if self.ship_hp <= 0:
                reward = -100 # Terminal penalty for destruction
            else: # Max steps reached
                reward = 100 # Terminal reward for survival
            if self.score > self.high_score:
                self.high_score = self.score

        self.score += reward

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_build_phase(self, movement, space_press, shift_press):
        # Handle menu navigation (debounced)
        if movement in [1, 2] and self.steps % 5 == 0: # Up/Down
            if movement == 1: # Up
                self.build_menu_index = (self.build_menu_index - 1) % len(self.build_options)
            elif movement == 2: # Down
                self.build_menu_index = (self.build_menu_index + 1) % len(self.build_options)
        
        # Handle actions (debounced)
        if (space_press or shift_press) and self.steps % 10 == 0:
            selected_option = self.build_options[self.build_menu_index]['label']

            if shift_press: # Shortcut to clone pilot
                self.pilots += 1
                self._create_simple_particle_burst(
                    (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), 20, self.COLOR_PLAYER)
                return 2 # Small reward for cloning

            if space_press:
                if selected_option == "Build Shield Matrix":
                    if self.pilots > 1: # Cost is 1 pilot
                        self.pilots -= 1
                        self.ship_parts += 1
                        self.max_ship_hp += 25
                        self.ship_hp = self.max_ship_hp # Full heal on upgrade
                        self._create_simple_particle_burst(
                            (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), 30, self.COLOR_SUCCESS)
                        return 5 # Reward for building
                    else:
                        self.screen_shake = 5
                        return -1 # Penalty for failed action
                elif selected_option == "Engage Cosmic Storm":
                    self._start_storm()
        return 0

    def _handle_storm_phase(self, movement):
        reward = 0
        hit_zone_y = self.SCREEN_HEIGHT - 80
        hit_window = 20 # pixels

        # Store player input for visual feedback
        if movement > 0:
            self.player_input_history.append({'type': movement, 'life': 10})
        
        # Update note positions
        for i in range(len(self.note_y_pos)):
            self.note_y_pos[i] += self.note_speed

        # Check for hits/misses
        if self.storm_progress_index < len(self.storm_sequence):
            current_note_type = self.storm_sequence[self.storm_progress_index]
            current_note_y = self.note_y_pos[self.storm_progress_index]

            # Check if a note is in the hit zone
            if abs(current_note_y - hit_zone_y) < hit_window:
                if movement == current_note_type:
                    reward += 1
                    self.note_hit_feedback[self.storm_progress_index] = 1 # Mark as hit
                    self._create_hit_particles(current_note_type, self.COLOR_SUCCESS)
                    self.storm_progress_index += 1
            
            # Check for missed notes that have passed the hit zone
            elif current_note_y > hit_zone_y + hit_window:
                reward -= 1
                self.ship_hp -= 10
                self.last_damage_time = self.steps
                self.screen_shake = 10
                self.note_hit_feedback[self.storm_progress_index] = -1 # Mark as missed
                self._create_hit_particles(current_note_type, self.COLOR_FAIL)
                self.storm_progress_index += 1

        # Update player input visual feedback
        self.player_input_history = [p for p in self.player_input_history if p['life'] > 0]
        for p in self.player_input_history:
            p['life'] -= 1

        # Check for storm completion
        if self.storm_progress_index >= len(self.storm_sequence):
            reward += 5 # Bonus for completing the storm
            self.successful_storms += 1
            self.ship_hp = min(self.max_ship_hp, self.ship_hp + 20) # Repair some damage
            
            # Difficulty scaling
            if self.successful_storms % 5 == 0:
                self.storm_level += 1
            
            # Unlock rewards
            if self.successful_storms % 10 == 0 and self.successful_storms > 0:
                reward += 10

            self.game_phase = 'build'
        
        return reward

    def _start_storm(self):
        self.game_phase = 'storm'
        self.storm_sequence = [self.np_random.integers(1, 5) for _ in range(self.storm_level)]
        self.note_y_pos = [-i * 100 for i in range(len(self.storm_sequence))]
        self.storm_progress_index = 0
        self.note_hit_feedback = [0] * len(self.storm_sequence)
        self.player_input_history.clear()

    def _check_termination(self):
        return self.ship_hp <= 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _create_simple_particle_burst(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 31),
                'color': color
            })

    def _create_hit_particles(self, note_type, color):
        x_pos = (self.SCREEN_WIDTH // 5) * note_type
        y_pos = self.SCREEN_HEIGHT - 80
        self._create_simple_particle_burst((x_pos, y_pos), 20, color)

    def _get_observation(self):
        # --- Main Rendering Function ---
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_ship()

        if self.game_phase == 'build':
            self._render_build_ui()
        elif self.game_phase == 'storm':
            self._render_storm_ui()

        self._render_particles()
        self._render_hud()

        # Apply screen shake
        render_surface = self.screen
        if self.screen_shake > 0:
            shake_offset_x = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            shake_offset_y = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            temp_surf.fill(self.COLOR_BG)
            temp_surf.blit(self.screen, (shake_offset_x, shake_offset_y))
            render_surface = temp_surf

        # Convert to numpy array
        arr = pygame.surfarray.array3d(render_surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, font, color, center_pos, antialias=True):
        rendered_text = font.render(text, antialias, color)
        text_rect = rendered_text.get_rect(center=center_pos)
        self.screen.blit(rendered_text, text_rect)

    def _render_background(self):
        # Static stars
        for x, y, size in self.stars:
            twinkle = self.np_random.integers(150, 256)
            pygame.draw.rect(self.screen, (twinkle, twinkle, twinkle), (x, y, size, size))
        
        # Slow-scrolling nebula
        for i, (x, y, r) in enumerate(self.nebula_particles):
            color = self.COLOR_NEBULA_1 if i % 2 == 0 else self.COLOR_NEBULA_2
            new_y = (y + self.steps * 0.2) % (self.SCREEN_HEIGHT + r*2) - r
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(new_y), r, color)

    def _render_ship(self):
        ship_center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        
        # Damage flash
        if self.steps - self.last_damage_time < 10:
            flash_alpha = 150 - (self.steps - self.last_damage_time) * 15
            pygame.gfxdraw.filled_circle(self.screen, ship_center[0], ship_center[1], 60, (*self.COLOR_FAIL, flash_alpha))

        # Main hull
        pygame.gfxdraw.filled_circle(self.screen, ship_center[0], ship_center[1], 40, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, ship_center[0], ship_center[1], 40, self.COLOR_PLAYER)
        
        # Ship parts
        for i in range(self.ship_parts):
            angle = i * (2 * math.pi / max(1, self.ship_parts)) + (self.steps / 50.0)
            px = ship_center[0] + math.cos(angle) * 55
            py = ship_center[1] + math.sin(angle) * 55
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), 10, self.COLOR_SUCCESS)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), 10, self.COLOR_SUCCESS)

    def _render_build_ui(self):
        self._render_text("BUILD PHASE", self.font_large, self.COLOR_UI_ACCENT, (self.SCREEN_WIDTH // 2, 80))
        
        for i, option in enumerate(self.build_options):
            y_pos = 200 + i * 50
            color = self.COLOR_UI_ACCENT if i == self.build_menu_index else self.COLOR_UI_TEXT
            
            label = option['label']
            if option['cost'] > 0:
                label += f" (Cost: {option['cost']} {option['resource']})"
            
            self._render_text(label, self.font_medium, color, (self.SCREEN_WIDTH // 2, y_pos))
            if i == self.build_menu_index:
                pygame.draw.line(self.screen, color, (self.SCREEN_WIDTH // 2 - 150, y_pos + 20), (self.SCREEN_WIDTH // 2 + 150, y_pos + 20), 2)
        
        self._render_text("[Arrows] Navigate | [Space] Select | [Shift] Clone Pilot", self.font_small, self.COLOR_UI_TEXT, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30))

    def _render_storm_ui(self):
        hit_zone_y = self.SCREEN_HEIGHT - 80
        
        # Draw hit zone targets
        for i in range(1, 5):
            x_pos = (self.SCREEN_WIDTH // 5) * i
            color = self.NOTE_COLORS[i-1]
            pygame.gfxdraw.aacircle(self.screen, x_pos, hit_zone_y, 20, color)
            
            # Draw player input feedback
            for p_input in self.player_input_history:
                if p_input['type'] == i:
                    alpha = int(255 * (p_input['life'] / 10))
                    pygame.gfxdraw.filled_circle(self.screen, x_pos, hit_zone_y, 20, (*color, alpha))

        # Draw falling notes
        for i in range(len(self.storm_sequence)):
            if self.note_hit_feedback[i] == 0: # Only draw pending notes
                note_type = self.storm_sequence[i]
                x_pos = (self.SCREEN_WIDTH // 5) * note_type
                y_pos = int(self.note_y_pos[i])
                color = self.NOTE_COLORS[note_type - 1]
                
                if 0 < y_pos < self.SCREEN_HEIGHT:
                    pygame.gfxdraw.filled_circle(self.screen, x_pos, y_pos, 15, color)
                    pygame.gfxdraw.aacircle(self.screen, x_pos, y_pos, 15, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 30.0))
            color_with_alpha = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['life'] / 5 + 1), color_with_alpha)

    def _render_hud(self):
        # Score and High Score
        self._render_text(f"SCORE: {int(self.score)}", self.font_medium, self.COLOR_UI_TEXT, (120, 30))
        self._render_text(f"HI: {int(self.high_score)}", self.font_small, self.COLOR_UI_TEXT, (120, 55))

        # Resources
        self._render_text(f"PILOTS: {self.pilots}", self.font_medium, self.COLOR_PLAYER, (self.SCREEN_WIDTH - 120, 30))
        self._render_text(f"PARTS: {self.ship_parts}", self.font_medium, self.COLOR_SUCCESS, (self.SCREEN_WIDTH - 120, 55))
        
        # Storm Level
        self._render_text(f"STORM LVL: {self.storm_level - 2}", self.font_small, self.COLOR_UI_ACCENT, (self.SCREEN_WIDTH // 2, 30))

        # HP Bar
        hp_ratio = max(0, self.ship_hp / self.max_ship_hp)
        hp_bar_width = 200
        hp_bar_height = 15
        hp_bar_x = self.SCREEN_WIDTH // 2 - hp_bar_width // 2
        hp_bar_y = self.SCREEN_HEIGHT - 35
        
        pygame.draw.rect(self.screen, self.COLOR_FAIL, (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_SUCCESS, (hp_bar_x, hp_bar_y, int(hp_bar_width * hp_ratio), hp_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height), 2)
        self._render_text(f"SHIELD: {self.ship_hp}/{self.max_ship_hp}", self.font_small, self.COLOR_UI_TEXT, (self.SCREEN_WIDTH // 2, hp_bar_y - 12))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ship_hp": self.ship_hp,
            "max_ship_hp": self.max_ship_hp,
            "pilots": self.pilots,
            "ship_parts": self.ship_parts,
            "successful_storms": self.successful_storms,
            "game_phase": self.game_phase,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Testing ---
    # This block will not be run by the evaluation server, but is useful for testing.
    # It requires pygame to be installed with a display driver.
    # To run, unset the SDL_VIDEODRIVER dummy variable, e.g.:
    # unset SDL_VIDEODRIVER
    
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Skipping manual test: requires a display driver.")
        exit()

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Cosmic Rhapsody - Manual Test")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Action Mapping for Human ---
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space_held = 0 # 0=released, 1=held
        shift_held = 0 # 0=released, 1=held

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Render to Screen ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {info['steps']}")
    env.close()