import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:33:59.393498
# Source Brief: brief_02909.md
# Brief Index: 2909
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Jump through historical eras while staying incognito to avoid witnesses. "
        "Move swiftly to build momentum and reach the future before you are discovered."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press Shift to toggle between "
        "Incognito (safe) and Public (dangerous) modes."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    ERA_WIDTH = 1280  # Virtual width of an era before transitioning

    # Player settings
    PLAYER_SPEED = 3.5
    PLAYER_FRICTION = 0.92
    PLAYER_SIZE = 12
    MOMENTUM_MAX = 100
    MOMENTUM_GAIN = 1.5
    MOMENTUM_DECAY = 0.5

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (40, 50, 70, 150)
    COLOR_INCOGNITO = (50, 200, 100)
    COLOR_PUBLIC = (255, 80, 80)
    COLOR_WITNESS = (255, 220, 50)

    # Era Definitions
    ERAS = [
        {
            "name": "Ancient Egypt",
            "bg_color": (45, 35, 25),
            "fg_color": (30, 20, 10),
            "witness_count": 1,
            "witness_speed": 0.8,
            "bg_elements": [ # type, x_ratio, y_ratio, size_ratio
                ("pyramid", 0.3, 0.8, 0.3),
                ("pyramid", 0.7, 0.8, 0.4)
            ]
        },
        {
            "name": "Medieval Europe",
            "bg_color": (40, 60, 40),
            "fg_color": (25, 45, 25),
            "witness_count": 2,
            "witness_speed": 1.0,
            "bg_elements": [
                ("castle", 0.5, 0.7, 0.5)
            ]
        },
        {
            "name": "Renaissance Italy",
            "bg_color": (70, 60, 80),
            "fg_color": (50, 40, 60),
            "witness_count": 3,
            "witness_speed": 1.2,
            "bg_elements": [
                ("aqueduct", 0.1, 0.7, 0.8),
                ("aqueduct", 0.9, 0.7, 0.8)
            ]
        },
        {
            "name": "The Future", # Victory Era
            "bg_color": (20, 20, 50),
            "fg_color": (10, 10, 30),
            "witness_count": 0,
            "witness_speed": 0,
            "bg_elements": []
        }
    ]

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
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_era = pygame.font.SysFont("serif", 28, bold=True)
        self.font_state = pygame.font.SysFont("monospace", 14, bold=True)

        # --- Game State Initialization ---
        self.player_pos = None
        self.player_vel = None
        self.is_public = None
        self.prev_shift_state = None
        self.momentum = None
        self.witnesses = []
        self.particles = []
        self.current_era_index = None
        self.world_scroll_x = None
        self.steps = None
        self.score = None
        self.game_over = None

        # self.reset() # reset() is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Player state
        self.player_pos = np.array([100.0, self.SCREEN_HEIGHT / 2.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_public = False
        self.prev_shift_state = False
        self.momentum = 0.0

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_era_index = 0
        self.world_scroll_x = 0.0
        
        # Entities
        self.particles = []
        self._spawn_witnesses()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0.0

        # --- Action Handling ---
        movement, _, shift_held = action[0], action[1] == 1, action[2] == 1

        # State switching (on press, not hold)
        if shift_held and not self.prev_shift_state:
            self.is_public = not self.is_public
            # Sound: state_switch.wav
        self.prev_shift_state = shift_held

        # --- Player Movement ---
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1.0 # Up
        elif movement == 2: move_vec[1] = 1.0 # Down
        elif movement == 3: move_vec[0] = -1.0 # Left
        elif movement == 4: move_vec[0] = 1.0 # Right

        if np.linalg.norm(move_vec) > 0:
            self.player_vel += move_vec * self.PLAYER_SPEED * 0.2 # Acceleration
            self.momentum = min(self.MOMENTUM_MAX, self.momentum + self.MOMENTUM_GAIN)
            if self.steps % 3 == 0: self._spawn_particles(1)
        else:
            self.momentum = max(0, self.momentum - self.MOMENTUM_DECAY)

        self.player_vel *= self.PLAYER_FRICTION
        if np.linalg.norm(self.player_vel) < 0.1: self.player_vel[:] = 0
        
        self.player_pos += self.player_vel
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        # --- World Scrolling ---
        if self.player_pos[0] > self.SCREEN_WIDTH * 0.6 and self.player_vel[0] > 0:
            scroll_amount = self.player_vel[0]
            self.world_scroll_x += scroll_amount
            self.player_pos[0] -= scroll_amount
            for w in self.witnesses:
                w['pos'][0] -= scroll_amount
            for p in self.particles:
                p['pos'][0] -= scroll_amount

        # --- Update Entities ---
        self._update_particles()
        self._update_witnesses()

        # --- Check Game State ---
        terminated = False
        
        # Detection
        if self.is_public:
            for w in self.witnesses:
                dist = np.linalg.norm(self.player_pos - w['pos'])
                if dist < w['detection_radius']:
                    terminated = True
                    reward = -100.0
                    self.game_over = True
                    # Sound: detection_fail.wav
                    break
        
        # Era Progression
        if self.world_scroll_x >= self.ERA_WIDTH:
            reward += 10.0
            self.score += 10
            self.current_era_index += 1
            self.world_scroll_x = 0
            if self.current_era_index >= len(self.ERAS) -1: # Reached final era
                terminated = True
                reward += 100.0
                self.score += 100
                self.game_over = True
                # Sound: victory.wav
            else:
                self._spawn_witnesses()
                # Sound: era_transition.wav
        
        # Max steps
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # In Gym API v26, truncated also means terminated

        # --- Calculate Reward ---
        if not terminated:
            reward += 0.1  # Survival reward
            reward += (self.momentum / self.MOMENTUM_MAX) * 0.1 # Momentum reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        era_info = self.ERAS[self.current_era_index]
        self.screen.fill(era_info["bg_color"])
        
        self._render_background(era_info)
        self._render_particles()
        self._render_witnesses()
        self._render_player()
        self._render_ui(era_info)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "era": self.ERAS[self.current_era_index]["name"],
            "momentum": self.momentum,
            "is_public": self.is_public
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    # --- Rendering Helpers ---
    def _render_background(self, era_info):
        for element in era_info["bg_elements"]:
            element_type, x_ratio, y_ratio, size_ratio = element
            x = (x_ratio * self.ERA_WIDTH - self.world_scroll_x) % (self.ERA_WIDTH + self.SCREEN_WIDTH) - self.SCREEN_WIDTH * 0.5
            y = y_ratio * self.SCREEN_HEIGHT
            
            if element_type == "pyramid":
                size = 200 * size_ratio
                points = [(x, y), (x + size, y), (x + size/2, y - size * 0.8)]
                pygame.draw.polygon(self.screen, era_info["fg_color"], points)
            elif element_type == "castle":
                w, h = 250 * size_ratio, 150 * size_ratio
                pygame.draw.rect(self.screen, era_info["fg_color"], (x - w/2, y - h, w, h))
                for i in range(5):
                    tw = w / 5
                    pygame.draw.rect(self.screen, era_info["fg_color"], (x - w/2 + i*tw, y-h-20, tw*0.6, 20))
            elif element_type == "aqueduct":
                 w, h = 150 * size_ratio, 80 * size_ratio
                 for i in range(-2, 10):
                     pygame.draw.rect(self.screen, era_info["fg_color"], (x + i * w, y - h, w, 20))
                     pygame.draw.rect(self.screen, era_info["fg_color"], (x + i * w, y - h, 20, h))
                     pygame.draw.rect(self.screen, era_info["fg_color"], (x + i * w + w - 20, y - h, 20, h))


    def _render_player(self):
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        player_color = self.COLOR_PUBLIC if self.is_public else self.COLOR_INCOGNITO
        
        # Glow effect
        for i in range(self.PLAYER_SIZE, 0, -2):
            alpha = 80 - (i * (80 / self.PLAYER_SIZE))
            glow_color = (*player_color, int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, px, py, i, glow_color)
        
        # Core
        pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_SIZE // 2, player_color)
        pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_SIZE // 2, player_color)

    def _render_witnesses(self):
        for w in self.witnesses:
            wx, wy = int(w['pos'][0]), int(w['pos'][1])
            
            # Body
            pygame.gfxdraw.filled_circle(self.screen, wx, wy, w['size'], self.COLOR_WITNESS)
            
            # Detection radius visualization (only if player is public)
            if self.is_public:
                radius = int(w['detection_radius'])
                # Pulsing effect
                pulse = abs(math.sin(self.steps * 0.1))
                alpha = int(30 + pulse * 40)
                pygame.gfxdraw.filled_circle(self.screen, wx, wy, radius, (*self.COLOR_WITNESS, alpha))
                pygame.gfxdraw.aacircle(self.screen, wx, wy, radius, (*self.COLOR_WITNESS, alpha + 30))

    def _render_particles(self):
        for p in self.particles:
            color = (*self.COLOR_INCOGNITO, p['alpha']) if not self.is_public else (*self.COLOR_PUBLIC, p['alpha'])
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

    def _render_ui(self, era_info):
        # Semi-transparent background for UI
        ui_bar = pygame.Surface((self.SCREEN_WIDTH, 80), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, 0))
        self.screen.blit(ui_bar, (0, self.SCREEN_HEIGHT - 30))

        # Top Bar
        era_text = self.font_era.render(f"ERA: {era_info['name']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(era_text, (10, 10))

        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Timeline
        timeline_y = 50
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (10, timeline_y), (self.SCREEN_WIDTH - 10, timeline_y), 2)
        progress_ratio = self.current_era_index / (len(self.ERAS) - 1) + (self.world_scroll_x / self.ERA_WIDTH) / (len(self.ERAS) - 1)
        progress_x = 10 + progress_ratio * (self.SCREEN_WIDTH - 20)
        pygame.draw.circle(self.screen, self.COLOR_INCOGNITO, (int(progress_x), timeline_y), 8)

        # Bottom Bar
        state_str = "PUBLIC" if self.is_public else "INCOGNITO"
        state_color = self.COLOR_PUBLIC if self.is_public else self.COLOR_INCOGNITO
        state_text = self.font_state.render(f"MODE: {state_str}", True, state_color)
        self.screen.blit(state_text, (10, self.SCREEN_HEIGHT - 22))

        # Momentum Bar
        momentum_w = 200
        momentum_h = 12
        momentum_x = self.SCREEN_WIDTH - momentum_w - 10
        momentum_y = self.SCREEN_HEIGHT - 22
        bar_fill = self.momentum / self.MOMENTUM_MAX
        pygame.draw.rect(self.screen, (60, 70, 90), (momentum_x, momentum_y, momentum_w, momentum_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (momentum_x, momentum_y, int(momentum_w * bar_fill), momentum_h))


    # --- Logic Helpers ---
    def _spawn_witnesses(self):
        self.witnesses.clear()
        era = self.ERAS[self.current_era_index]
        for _ in range(era["witness_count"]):
            patrol_y = self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            patrol_start_x = self.np_random.uniform(100, self.SCREEN_WIDTH - 100)
            patrol_end_x = self.np_random.uniform(100, self.SCREEN_WIDTH - 100)
            self.witnesses.append({
                "pos": np.array([patrol_start_x, patrol_y]),
                "start": patrol_start_x,
                "end": patrol_end_x,
                "speed": era["witness_speed"] * (1 if patrol_end_x > patrol_start_x else -1),
                "size": 8,
                "detection_radius": self.np_random.uniform(60, 90)
            })

    def _update_witnesses(self):
        for w in self.witnesses:
            w['pos'][0] += w['speed']
            if w['speed'] > 0 and w['pos'][0] > w['end']: w['speed'] *= -1
            if w['speed'] < 0 and w['pos'][0] < w['start']: w['speed'] *= -1
            # Swap start/end if needed to keep logic simple
            if w['start'] > w['end']: w['start'], w['end'] = w['end'], w['start']

    def _spawn_particles(self, count):
        for _ in range(count):
            self.particles.append({
                'pos': self.player_pos.copy() - self.player_vel * 2,
                'vel': self.np_random.uniform(-0.5, 0.5, size=2) - self.player_vel * 0.1,
                'size': self.np_random.uniform(2, 5),
                'life': self.np_random.uniform(20, 40),
                'alpha': 255
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] -= 0.05
            p['alpha'] = max(0, int(255 * (p['life'] / 40)))
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]

if __name__ == "__main__":
    # --- Manual Play Example ---
    # Re-enable display for human playing
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human playing
    pygame.display.set_caption("History Jumper")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Convert observation from (H, W, C) to (W, H, C) for pygame display
    surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    while not terminated and not truncated:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the new state
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        print(f"Step: {info['steps']}, Era: {info['era']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        
        clock.tick(30) # Run at 30 FPS

    print("\n--- GAME OVER ---")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {info['steps']}")
    env.close()