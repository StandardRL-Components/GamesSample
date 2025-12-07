import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:02:43.591964
# Source Brief: brief_00775.md
# Brief Index: 775
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Harmonic Resonance: A Gymnasium environment where the agent places sound-wave-emitting
    cards to match a target resonance frequency. The agent controls a cursor to place
    cards and can unlock and use special abilities to manipulate the waves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Place sound-wave-emitting cards to match a target resonance frequency. "
        "Unlock special abilities to manipulate the waves and achieve harmonic synchrony."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. "
        "Press space to place a card or use an ability. Press shift to cycle through available tools."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 80
    GAME_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT
    FPS = 30
    MAX_STEPS = 1200 # 40 seconds at 30 FPS

    # Colors
    COLOR_BG = (15, 18, 28)
    COLOR_GRID = (30, 35, 50)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_TARGET_FREQ = (255, 220, 0) # Yellow

    # Game Parameters
    CURSOR_SPEED = 10
    WAVE_SPEED = 2.0
    WAVE_AMPLITUDE_DECAY = 0.99
    WIN_DURATION_STEPS = 30 # Must hold resonance for 1 second

    # Card Frequencies and Colors
    CARD_FREQUENCIES = [50, 100, 150]
    CARD_COLORS = [(255, 50, 50), (50, 255, 50), (80, 80, 255)] # Red, Green, Blue

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        # actions[1]: Space button (0=released, 1=held) -> Place/Use
        # actions[2]: Shift button (0=released, 1=held) -> Cycle Tool
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        # Persistent state (across resets)
        self.total_wins = 0
        self.unlocked_abilities = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.resonance_hold_timer = 0
        
        # Game objects
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.GAME_HEIGHT / 2], dtype=float)
        self.cards = [] # {pos, freq, color}
        self.waves = [] # {pos, freq, color, radius, amplitude}
        self.particles = [] # {pos, vel, life, color}
        
        # Player state
        self.selected_tool_idx = 0
        self.available_tools = self._get_available_tools()
        
        # Input state for press detection
        self.prev_space_held = False
        self.prev_shift_held = False

        # Level setup
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        if not self.game_over:
            self._handle_input(movement, space_held, shift_held)
            self._update_game_state()
            
            reward = self._calculate_reward()
            self.score += reward
            
            terminated = self._check_termination()
            if terminated and self.win:
                self.total_wins += 1
                if self._check_unlocks():
                    reward += 50.0 # Bonus for unlocking a new ability
                    self.score += 50.0
                    self.available_tools = self._get_available_tools()

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GAME_HEIGHT - 1)

        # --- Cycle Tool (on Shift press) ---
        is_shift_press = shift_held and not self.prev_shift_held
        if is_shift_press:
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.available_tools)

        # --- Place/Use Tool (on Space press) ---
        is_space_press = space_held and not self.prev_space_held
        if is_space_press:
            tool_name = self.available_tools[self.selected_tool_idx]
            if "Card" in tool_name:
                self._place_card()
            elif "Boost" in tool_name:
                self._use_amplitude_boost()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_state(self):
        self.steps += 1

        # --- Update Waves ---
        new_waves = []
        for wave in self.waves:
            wave['radius'] += self.WAVE_SPEED
            wave['amplitude'] *= self.WAVE_AMPLITUDE_DECAY
            if wave['amplitude'] > 0.01 and wave['radius'] < self.SCREEN_WIDTH:
                new_waves.append(wave)
        self.waves = new_waves

        # --- Update Particles ---
        new_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

        # --- Calculate Resonance ---
        self.current_resonance = self._calculate_resonance()

        # --- Check Win Condition ---
        if abs(self.current_resonance - self.target_frequency) < 5.0:
            self.resonance_hold_timer += 1
        else:
            self.resonance_hold_timer = 0
        
        if self.resonance_hold_timer >= self.WIN_DURATION_STEPS:
            self.game_over = True
            self.win = True

    def _calculate_reward(self):
        if self.win:
            return 100.0

        if self.steps >= self.MAX_STEPS:
            return -100.0
        
        freq_diff = abs(self.current_resonance - self.target_frequency)
        reward = max(0, 1.0 - (freq_diff / 100.0)) * 0.1
        
        return reward

    def _check_termination(self):
        return self.game_over
    
    def _setup_level(self):
        # Difficulty scales with total wins
        if self.total_wins < 5:
            self.target_frequency = self.np_random.choice(self.CARD_FREQUENCIES)
        elif self.total_wins < 10:
            f1, f2 = self.np_random.choice(self.CARD_FREQUENCIES, 2, replace=False)
            self.target_frequency = (f1 + f2) / 2.0
        else:
            self.target_frequency = sum(self.CARD_FREQUENCIES) / 3.0
            
        self.sensor_points = [
            np.array([self.SCREEN_WIDTH * 0.25, self.GAME_HEIGHT / 2]),
            np.array([self.SCREEN_WIDTH * 0.75, self.GAME_HEIGHT / 2])
        ]
        self.current_resonance = 0

    def _get_available_tools(self):
        tools = [f"Card {i+1}" for i in range(len(self.CARD_FREQUENCIES))]
        tools.extend(self.unlocked_abilities)
        return tools

    def _check_unlocks(self):
        unlocked_something = False
        if self.total_wins >= 2 and "Amplitude Boost" not in self.unlocked_abilities:
            self.unlocked_abilities.append("Amplitude Boost")
            unlocked_something = True
        return unlocked_something

    def _place_card(self):
        card_idx = self.selected_tool_idx
        if card_idx < len(self.CARD_FREQUENCIES):
            new_card = {
                'pos': self.cursor_pos.copy(),
                'freq': self.CARD_FREQUENCIES[card_idx],
                'color': self.CARD_COLORS[card_idx]
            }
            self.cards.append(new_card)
            
            self.waves.append({
                'pos': new_card['pos'],
                'freq': new_card['freq'],
                'color': new_card['color'],
                'radius': 0,
                'amplitude': 1.0
            })
            self._create_particles(self.cursor_pos, new_card['color'], 20)

    def _use_amplitude_boost(self):
        for wave in self.waves:
            dist = np.linalg.norm(wave['pos'] - self.cursor_pos)
            if dist < 50:
                wave['amplitude'] = min(1.5, wave['amplitude'] * 2.0)
        self._create_particles(self.cursor_pos, (255, 255, 100), 30, is_burst=True)

    def _calculate_resonance(self):
        total_amplitude = 0
        weighted_freq_sum = 0
        
        if not self.waves:
            return 0

        for sensor in self.sensor_points:
            for wave in self.waves:
                dist_to_sensor = np.linalg.norm(wave['pos'] - sensor)
                if abs(dist_to_sensor - wave['radius']) < self.WAVE_SPEED * 2:
                    hit_amplitude = wave['amplitude'] * max(0, 1 - (abs(dist_to_sensor - wave['radius']) / (self.WAVE_SPEED * 2)))
                    weighted_freq_sum += wave['freq'] * hit_amplitude
                    total_amplitude += hit_amplitude
        
        if total_amplitude == 0:
            return 0
            
        return weighted_freq_sum / total_amplitude

    def _create_particles(self, pos, color, count, is_burst=False):
        for _ in range(count):
            if is_burst:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            else:
                vel = self.np_random.uniform(-1, 1, size=2) * 2
            
            self.particles.append({
                'pos': pos.copy() + self.np_random.uniform(-5, 5, size=2),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.GAME_HEIGHT))
        for y in range(0, self.GAME_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        
        for card in self.cards:
            pos = (int(card['pos'][0]), int(card['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, card['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, card['color'])

        for wave in self.waves:
            pos = (int(wave['pos'][0]), int(wave['pos'][1]))
            radius = int(wave['radius'])
            alpha = int(wave['amplitude'] * 200)
            if radius > 0 and alpha > 10:
                color = (*wave['color'], alpha)
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius-1, color)
                self.screen.blit(temp_surf, (pos[0]-radius, pos[1]-radius))

        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['life'] / 10)
            if size > 0:
                alpha = int((p['life'] / 40) * 255)
                color_with_alpha = (*p['color'], alpha)
                temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                temp_surf.fill(color_with_alpha)
                self.screen.blit(temp_surf, pos)

        for sensor in self.sensor_points:
            pos = (int(sensor[0]), int(sensor[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_TARGET_FREQ)

        c_pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        tool_color = self.CARD_COLORS[self.selected_tool_idx] if "Card" in self.available_tools[self.selected_tool_idx] else (255, 255, 100)
        pygame.draw.line(self.screen, tool_color, (c_pos[0] - 10, c_pos[1]), (c_pos[0] + 10, c_pos[1]), 2)
        pygame.draw.line(self.screen, tool_color, (c_pos[0], c_pos[1] - 10), (c_pos[0], c_pos[1] + 10), 2)
        pygame.gfxdraw.aacircle(self.screen, c_pos[0], c_pos[1], 12, tool_color)

    def _render_ui(self):
        ui_rect = pygame.Rect(0, self.GAME_HEIGHT, self.SCREEN_WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, (25, 30, 45), ui_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.GAME_HEIGHT), (self.SCREEN_WIDTH, self.GAME_HEIGHT), 2)

        def draw_text(text, font, color, pos, center=False):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect()
            if center: text_rect.center = pos
            else: text_rect.topleft = pos
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)

        meter_x, meter_y, meter_w, meter_h = 20, self.GAME_HEIGHT + 15, 200, 50
        max_freq = max(self.CARD_FREQUENCIES)
        
        pygame.draw.rect(self.screen, self.COLOR_BG, (meter_x, meter_y, meter_w, meter_h))
        
        current_pos = int((self.current_resonance / max_freq) * meter_w)
        current_pos = np.clip(current_pos, 0, meter_w)
        pygame.draw.rect(self.screen, (150, 150, 255), (meter_x, meter_y, current_pos, meter_h))
        
        target_pos = int((self.target_frequency / max_freq) * meter_w)
        pygame.draw.line(self.screen, self.COLOR_TARGET_FREQ, (meter_x + target_pos, meter_y), (meter_x + target_pos, meter_y + meter_h), 3)

        if self.resonance_hold_timer > 0:
            progress = self.resonance_hold_timer / self.WIN_DURATION_STEPS
            pygame.draw.rect(self.screen, self.COLOR_TARGET_FREQ, (meter_x + target_pos - 5, meter_y + meter_h, 10, -meter_h * progress), 0, border_radius=2)
            
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (meter_x, meter_y, meter_w, meter_h), 2, border_radius=3)
        draw_text(f"Resonance: {self.current_resonance:.1f} Hz", self.font_small, self.COLOR_TEXT, (meter_x, self.GAME_HEIGHT + 15 - 20))
        draw_text(f"Target: {self.target_frequency:.1f} Hz", self.font_small, self.COLOR_TARGET_FREQ, (meter_x + 90, self.GAME_HEIGHT + 15 - 20))

        tool_x = 260
        draw_text("Selected Tool:", self.font_small, self.COLOR_TEXT, (tool_x, self.GAME_HEIGHT + 15))
        tool_name = self.available_tools[self.selected_tool_idx]
        tool_color = self.COLOR_TEXT
        if "Card" in tool_name:
            tool_color = self.CARD_COLORS[self.selected_tool_idx]
        elif "Boost" in tool_name:
            tool_color = (255, 255, 100)
        draw_text(tool_name, self.font_medium, tool_color, (tool_x, self.GAME_HEIGHT + 35))

        info_x = 480
        draw_text(f"Score: {self.score:.1f}", self.font_medium, self.COLOR_TEXT, (info_x, self.GAME_HEIGHT + 15))
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        draw_text(f"Time: {time_left:.1f}s", self.font_medium, self.COLOR_TEXT, (info_x, self.GAME_HEIGHT + 45))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wins": self.total_wins,
            "target_frequency": self.target_frequency,
            "current_resonance": self.current_resonance,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # To run and play the game manually, we need a display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Harmonic Resonance")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wins: {info['wins']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)
        
    env.close()