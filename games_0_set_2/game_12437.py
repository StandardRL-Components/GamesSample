import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:34:42.101992
# Source Brief: brief_02437.md
# Brief Index: 2437
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import math
import random

class GameEnv(gym.Env):
    """
    Tower Stacker: A Gymnasium environment where the agent stacks colored blocks
    to build stable towers and achieve a high score.

    The agent controls a preview block at the top of the screen, selecting its
    type and which of the three tower slots to drop it into. The goal is to
    build tall, stable towers to reach a score of 200.

    Visuals are a key focus, with smooth animations, particle effects, and a
    clean, geometric aesthetic to provide a satisfying user experience.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) -> Drops block on press
    - actions[2]: Shift button (0=released, 1=held) -> No effect
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack colored blocks to build stable towers. Score points by building tall and "
        "keeping the tower heights synchronized."
    )
    user_guide = (
        "Controls: ←→ to select a tower, ↑↓ to change the block color. Press space to drop the block."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        self.GROUND_Y = 360
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 50, 25
        self.TOWER_SLOTS_X = [self.WIDTH // 4, self.WIDTH // 2, self.WIDTH * 3 // 4]
        self.PREVIEW_Y = 60
        self.BLOCK_DROP_SPEED = 30
        self.MAX_STEPS = 2000
        self.WIN_SCORE = 200

        # --- Visuals & Colors ---
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_GROUND = (40, 50, 60)
        self.BLOCK_COLORS = {
            0: (230, 70, 70),   # Red
            1: (70, 230, 70),   # Green
            2: (70, 120, 230),  # Blue
        }
        self.BLOCK_SHADOW_COLORS = {k: tuple(int(c * 0.6) for c in v) for k, v in self.BLOCK_COLORS.items()}
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_WIN_TEXT = (255, 220, 100)

        # --- Fonts ---
        try:
            self.font_main = pygame.font.SysFont("Consolas", 22, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 28)
            self.font_large = pygame.font.SysFont(None, 60)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.towers = []
        self.preview_block = {}
        self.falling_blocks = []
        self.particles = []
        self.prev_space_held = False

        self.np_random = None # Will be seeded in reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Each tower is a list of block dictionaries
        self.towers = [[], [], []]
        self.falling_blocks = []
        self.particles = []

        self.preview_block = {
            'type': self.np_random.integers(0, 3),
            'tower_idx': 1  # Start in the middle
        }
        self.prev_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0

        self._handle_actions(action)
        step_reward += self._update_game_state()

        terminated = (self.score >= self.WIN_SCORE) or (self.steps >= self.MAX_STEPS)
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                step_reward += 100
            self.game_over = True
        
        truncated = False # This environment does not truncate based on time limits in the step.

        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Only allow action if no block is currently falling
        if not self.falling_blocks:
            if movement == 1:  # Up: Cycle block type forward
                self.preview_block['type'] = (self.preview_block['type'] + 1) % 3
            elif movement == 2:  # Down: Cycle block type backward
                self.preview_block['type'] = (self.preview_block['type'] - 1 + 3) % 3
            elif movement == 3:  # Left: Move selection left
                self.preview_block['tower_idx'] = max(0, self.preview_block['tower_idx'] - 1)
            elif movement == 4:  # Right: Move selection right
                self.preview_block['tower_idx'] = min(2, self.preview_block['tower_idx'] - 1)

            # Space press (0->1 transition) triggers block drop
            if space_held and not self.prev_space_held:
                self._drop_block()
                # sfx: block_initiate_drop.wav

        self.prev_space_held = space_held

    def _drop_block(self):
        tower_idx = self.preview_block['tower_idx']
        tower = self.towers[tower_idx]

        target_y = self.GROUND_Y - self.BLOCK_HEIGHT
        if tower:
            target_y = tower[-1]['y'] - self.BLOCK_HEIGHT

        new_block = {
            'x': self.TOWER_SLOTS_X[tower_idx],
            'y': self.PREVIEW_Y,
            'type': self.preview_block['type'],
            'w': self.BLOCK_WIDTH,
            'h': self.BLOCK_HEIGHT,
            'target_y': target_y,
            'tower_idx': tower_idx
        }
        self.falling_blocks.append(new_block)
        self.preview_block['type'] = self.np_random.integers(0, 3)

    def _update_game_state(self):
        reward = self._update_falling_blocks()
        self._update_particles()
        return reward

    def _update_falling_blocks(self):
        reward = 0
        for block in self.falling_blocks[:]:
            block['y'] += self.BLOCK_DROP_SPEED
            if block['y'] >= block['target_y']:
                block['y'] = block['target_y']
                self.towers[block['tower_idx']].append(block)
                self.falling_blocks.remove(block)
                
                # --- Landing Logic ---
                # sfx: block_land_heavy.wav
                self.score += 1
                reward += 1
                self._create_particles((block['x'], block['y'] + self.BLOCK_HEIGHT), self.BLOCK_COLORS[block['type']], 25)

                if not self._check_tower_stability(block['tower_idx']):
                    # sfx: tower_collapse_rumble.wav
                    self._collapse_tower(block['tower_idx'])
                elif self._check_height_sync():
                    reward += 25
                    # sfx: height_sync_chime.wav
        return reward

    def _check_tower_stability(self, tower_idx):
        tower = self.towers[tower_idx]
        if len(tower) <= 1:
            return True

        base_block = tower[0]
        base_left = base_block['x'] - base_block['w'] / 2
        base_right = base_block['x'] + base_block['w'] / 2

        com_x = sum(b['x'] for b in tower) / len(tower)
        return base_left <= com_x <= base_right

    def _collapse_tower(self, tower_idx):
        for block in self.towers[tower_idx]:
            self._create_particles((block['x'], block['y']), self.BLOCK_COLORS[block['type']], 15, is_collapse=True)
        self.towers[tower_idx] = []

    def _check_height_sync(self):
        heights = [len(t) for t in self.towers]
        if not any(heights):
            return False
        non_empty_heights = [h for h in heights if h > 0]
        if not non_empty_heights:
            return False
        return max(non_empty_heights) - min(non_empty_heights) <= 1

    def _create_particles(self, pos, color, count, is_collapse=False):
        for _ in range(count):
            if is_collapse:
                angle = self.np_random.uniform(math.pi, 2 * math.pi)
                speed = self.np_random.uniform(5, 13)
                life = self.np_random.uniform(30, 50)
            else: # Placement particles
                angle = self.np_random.uniform(-math.pi * 0.85, -math.pi * 0.15)
                speed = self.np_random.uniform(2, 7)
                life = self.np_random.uniform(20, 40)

            self.particles.append({
                'x': pos[0] + self.np_random.uniform(-5, 5),
                'y': pos[1] + self.np_random.uniform(-5, 5),
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'color': color, 'life': life, 'max_life': life,
            })

    def _update_particles(self):
        gravity = 0.35
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        
        for tower in self.towers:
            for block in tower:
                self._render_block(block)

        for block in self.falling_blocks:
            self._render_block(block)

        if not self.falling_blocks and not self.game_over:
            self._render_preview()
            
        self._render_particles()

    def _render_block(self, block):
        x, y = int(block['x'] - block['w'] / 2), int(block['y'])
        w, h = int(block['w']), int(block['h'])
        main_color = self.BLOCK_COLORS[block['type']]
        shadow_color = self.BLOCK_SHADOW_COLORS[block['type']]
        
        pygame.draw.rect(self.screen, shadow_color, (x, y, w, h))
        pygame.draw.rect(self.screen, main_color, (x + 2, y, w - 4, h - 2))

    def _render_preview(self):
        idx = self.preview_block['tower_idx']
        preview_x = self.TOWER_SLOTS_X[idx]
        preview_type = self.preview_block['type']
        
        # Draw dashed line to landing spot
        tower = self.towers[idx]
        target_y = self.GROUND_Y - self.BLOCK_HEIGHT if not tower else tower[-1]['y'] - self.BLOCK_HEIGHT
        start_y, end_y = self.PREVIEW_Y + self.BLOCK_HEIGHT / 2, target_y + self.BLOCK_HEIGHT / 2
        for y in range(int(start_y), int(end_y), 12):
            pygame.draw.line(self.screen, (200, 200, 200, 100), (preview_x, y), (preview_x, y + 6), 1)

        # Render transparent preview block
        s = pygame.Surface((self.BLOCK_WIDTH, self.BLOCK_HEIGHT), pygame.SRCALPHA)
        main_color_alpha = (*self.BLOCK_COLORS[preview_type], 120)
        shadow_color_alpha = (*self.BLOCK_SHADOW_COLORS[preview_type], 120)
        pygame.draw.rect(s, shadow_color_alpha, (0, 0, self.BLOCK_WIDTH, self.BLOCK_HEIGHT))
        pygame.draw.rect(s, main_color_alpha, (2, 0, self.BLOCK_WIDTH - 4, self.BLOCK_HEIGHT - 2))
        self.screen.blit(s, (preview_x - self.BLOCK_WIDTH / 2, self.PREVIEW_Y - self.BLOCK_HEIGHT / 2))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = max(1, int(6 * (p['life'] / p['max_life'])))
            temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, size, size))
            self.screen.blit(temp_surf, (int(p['x']), int(p['y'])))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))

        steps_text = self.font_main.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 15, 10))
        
        if self.game_over:
            win_text = "TARGET REACHED!" if self.score >= self.WIN_SCORE else "TIME UP"
            text_surf = self.font_large.render(win_text, True, self.COLOR_WIN_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block is for human play and visualization.
    # It is not part of the required Gymnasium interface.
    # We need to unset the dummy video driver to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    pygame.display.set_caption("Tower Stacker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # --- Manual Control Mapping ---
    # A/D or Left/Right Arrow: Move selection
    # W/S or Up/Down Arrow: Cycle block type
    # Space: Drop block
    
    action = [0, 0, 0] # [movement, space, shift]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset env
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_SPACE:
                    action[1] = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0
        
        # Continuous key presses for movement
        keys = pygame.key.get_pressed()
        action[0] = 0 # Default to no movement
        
        # These actions are mutually exclusive in the original logic,
        # but we handle them sequentially here. The last one pressed wins.
        # This is fine for human play.
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action[0] = 4

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # The observation is the rendered game frame.
        # We transpose it back to Pygame's (width, height, channel) format
        # and create a surface to display.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Run at 30 FPS

    env.close()