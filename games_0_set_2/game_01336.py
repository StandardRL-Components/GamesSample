
# Generated: 2025-08-27T16:49:28.338011
# Source Brief: brief_01336.md
# Brief Index: 1336

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move your character (white square). Push blocks into their matching colored targets."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Push all 5 colored blocks to their matching targets before the time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Configuration ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.TILE_SIZE = 40  # 400px height / 10 tiles
        self.ARENA_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.TILE_SIZE) // 2
        self.ARENA_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.TILE_SIZE) // 2
        
        self.MAX_STEPS = 1000
        self.NUM_BLOCKS = 5

        # --- Colors & Style ---
        self.COLOR_BG = (20, 20, 20)
        self.COLOR_FLOOR = (40, 40, 40)
        self.COLOR_PLAYER = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.TARGET_COLORS = [pygame.Color(c).lerp((0,0,0), 0.6) for c in self.BLOCK_COLORS]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 30, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18)
        
        # --- Game State (initialized in reset) ---
        self.player_pos = None
        self.blocks = None
        self.targets = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.np_random = None
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        num_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        num_entities = 1 + self.NUM_BLOCKS + self.NUM_BLOCKS
        if num_cells < num_entities:
            raise ValueError("Grid too small for all entities")

        positions_1d = self.np_random.choice(num_cells, size=num_entities, replace=False)
        all_pos = [(i % self.GRID_WIDTH, i // self.GRID_WIDTH) for i in positions_1d]
        
        self.player_pos = all_pos.pop()
        
        target_positions = [all_pos.pop() for _ in range(self.NUM_BLOCKS)]
        block_positions = [all_pos.pop() for _ in range(self.NUM_BLOCKS)]

        self.targets = []
        for i in range(self.NUM_BLOCKS):
            self.targets.append({
                "pos": target_positions[i],
                "color": self.TARGET_COLORS[i],
                "id": i,
                "block_on": False
            })

        self.blocks = []
        for i in range(self.NUM_BLOCKS):
            self.blocks.append({
                "pos": block_positions[i],
                "color": self.BLOCK_COLORS[i],
                "id": i,
                "target_id": i
            })
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.1  # Penalty for taking a step

        self._update_particles()

        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
            next_player_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            if not (0 <= next_player_pos[0] < self.GRID_WIDTH and 0 <= next_player_pos[1] < self.GRID_HEIGHT):
                # Player bumps into wall
                self._create_collision_particles(self.player_pos, (dx, dy))
                # sound: bump_wall.wav
            else:
                pushed_block_idx = next((i for i, b in enumerate(self.blocks) if b["pos"] == next_player_pos), -1)
                
                if pushed_block_idx != -1:
                    reward += self._handle_block_push(pushed_block_idx, dx, dy)
                else:
                    self.player_pos = next_player_pos
        
        self.steps += 1
        
        terminated, win_lose_reward = self._check_termination()
        reward += win_lose_reward
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_block_push(self, block_idx, dx, dy):
        block = self.blocks[block_idx]
        next_block_pos = (block["pos"][0] + dx, block["pos"][1] + dy)
        push_reward = 0

        can_push = True
        if not (0 <= next_block_pos[0] < self.GRID_WIDTH and 0 <= next_block_pos[1] < self.GRID_HEIGHT):
            can_push = False # Cannot push into a wall
        elif any(b["pos"] == next_block_pos for b in self.blocks):
            can_push = False # Cannot push into another block
        
        if can_push:
            target = self.targets[block["target_id"]]
            is_on_target_before = (block["pos"] == target["pos"])
            
            old_dist = abs(block["pos"][0] - target["pos"][0]) + abs(block["pos"][1] - target["pos"][1])
            new_dist = abs(next_block_pos[0] - target["pos"][0]) + abs(next_block_pos[1] - target["pos"][1])
            
            push_reward += (old_dist - new_dist) # +1 for closer, -1 for further

            block["pos"] = next_block_pos
            self.player_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            is_on_target_after = (block["pos"] == target["pos"])

            if is_on_target_after and not is_on_target_before:
                push_reward += 10 # Event reward for placing block
                # sound: target_placed.wav
            elif not is_on_target_after and is_on_target_before:
                push_reward -= 10 # Penalty for moving block off target

            self._create_collision_particles(block["pos"], (-dx, -dy))
            # sound: block_push.wav
        else:
            # Push failed
            self._create_collision_particles(block["pos"], (dx, dy))
            # sound: bump_block.wav
        
        return push_reward

    def _check_termination(self):
        # Update which blocks are on targets
        for target in self.targets:
            target["block_on"] = any(b["id"] == target["id"] and b["pos"] == target["pos"] for b in self.blocks)
        
        win = all(t["block_on"] for t in self.targets)
        lose = self.steps >= self.MAX_STEPS
        
        if win:
            self.game_over = True
            # sound: win_jingle.wav
            return True, 50
        if lose:
            self.game_over = True
            # sound: lose_sound.wav
            return True, -50
            
        return False, 0
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        floor_rect = pygame.Rect(
            self.ARENA_X_OFFSET, self.ARENA_Y_OFFSET,
            self.GRID_WIDTH * self.TILE_SIZE, self.GRID_HEIGHT * self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, floor_rect)

        for target in self.targets:
            tx, ty = target["pos"]
            rect = pygame.Rect(
                self.ARENA_X_OFFSET + tx * self.TILE_SIZE, self.ARENA_Y_OFFSET + ty * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )
            pygame.draw.rect(self.screen, target["color"], rect)
            if target["block_on"]:
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 3)

        for block in self.blocks:
            bx, by = block["pos"]
            rect = pygame.Rect(
                self.ARENA_X_OFFSET + bx * self.TILE_SIZE + 4, self.ARENA_Y_OFFSET + by * self.TILE_SIZE + 4,
                self.TILE_SIZE - 8, self.TILE_SIZE - 8
            )
            pygame.draw.rect(self.screen, block["color"], rect, border_radius=4)
            border_color = pygame.Color(block["color"]).lerp((0,0,0), 0.4)
            pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=4)

        px, py = self.player_pos
        player_rect = pygame.Rect(
            self.ARENA_X_OFFSET + px * self.TILE_SIZE + 8, self.ARENA_Y_OFFSET + py * self.TILE_SIZE + 8,
            self.TILE_SIZE - 16, self.TILE_SIZE - 16
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        for p in self.particles:
            p_color = pygame.Color(p['color'])
            p_color.a = int(255 * (p['lifetime'] / p['max_lifetime']))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p_color)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 10))

        steps_left = self.MAX_STEPS - self.steps
        time_color = (255, 255, 255) if steps_left > 200 else (255, 100, 100)
        steps_text = self.font_large.render(f"TIME: {steps_left}", True, time_color)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 20, 10))
        
        blocks_on_target = sum(1 for t in self.targets if t["block_on"])
        blocks_text = self.font_small.render(f"BLOCKS: {blocks_on_target} / {self.NUM_BLOCKS}", True, (200, 200, 200))
        self.screen.blit(blocks_text, (self.SCREEN_WIDTH // 2 - blocks_text.get_width() // 2, self.SCREEN_HEIGHT - 30))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            msg = "LEVEL CLEAR!" if all(t["block_on"] for t in self.targets) else "TIME UP!"
            color = (100, 255, 100) if "CLEAR" in msg else (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(end_text, text_rect)
            
    def _create_collision_particles(self, grid_pos, direction_of_impact):
        px = self.ARENA_X_OFFSET + grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2
        py = self.ARENA_Y_OFFSET + grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2
        
        for _ in range(10):
            angle = math.atan2(-direction_of_impact[1], -direction_of_impact[0]) + self.np_random.uniform(-math.pi/3, math.pi/3)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 25)
            self.particles.append({
                'pos': [px, py], 'vel': vel,
                'radius': self.np_random.uniform(2, 4),
                'lifetime': lifetime, 'max_lifetime': lifetime,
                'color': (200, 200, 200)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['lifetime'] -= 1
            p['radius'] *= 0.95

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_on_target": sum(1 for t in self.targets if t["block_on"]),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Need to reset to initialize state before getting observation
        if self.player_pos is None:
            self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")