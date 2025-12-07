
# Generated: 2025-08-27T16:21:15.753536
# Source Brief: brief_01200.md
# Brief Index: 1200

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Hold Space and press an arrow key to push a block."
    )

    game_description = (
        "A minimalist puzzle game. Push colored blocks onto their matching targets before you run out of moves."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 10, 8
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_W * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_H * self.CELL_SIZE) // 2
        self.ANIMATION_SPEED = 0.2  # Higher is faster

        # --- Colors ---
        self.COLOR_BG = (34, 40, 49) # Dark blue-grey
        self.COLOR_GRID = (57, 62, 70) # Medium grey
        self.COLOR_SELECTOR = (255, 211, 105) # Yellow
        self.BLOCK_COLORS = [
            (0, 173, 181),   # Cyan
            (238, 238, 238), # White
            (255, 46, 99),   # Pink
            (129, 236, 236), # Light Cyan
            (255, 168, 168)  # Light Pink
        ]
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- State Variables ---
        self.rng = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.max_moves = 0
        self.blocks = []
        self.selector_pos = (0, 0)
        self.selector_visual_pos = [0.0, 0.0]
        self.particles = []
        self.is_animating = False
        self.level = 1
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Fallback to a default generator if no seed is provided
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Level generation
        self.num_blocks = min(3 + (self.level - 1) // 2, len(self.BLOCK_COLORS))
        self.max_moves = 20 + self.num_blocks * 5
        self.moves_left = self.max_moves

        # Generate unique positions for blocks and targets
        possible_positions = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.rng.shuffle(possible_positions)
        
        self.blocks = []
        used_colors = self.rng.choice(self.BLOCK_COLORS, self.num_blocks, replace=False).tolist()

        for i in range(self.num_blocks):
            block_pos = possible_positions.pop()
            target_pos = possible_positions.pop()
            color = tuple(used_colors[i])
            self.blocks.append({
                "id": i,
                "pos": block_pos,
                "visual_pos": [float(block_pos[0]), float(block_pos[1])],
                "target_pos": target_pos,
                "color": color,
                "is_on_target": False,
                "animation_progress": 1.0,
                "start_pos": block_pos
            })

        self.selector_pos = (0, 0)
        self.selector_visual_pos = [0.0, 0.0]
        self.particles = []
        self.is_animating = False
        
        self._check_initial_targets()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(60) # Run at 60fps for smoother animations
        self.steps += 1
        reward = 0

        # --- Update Animations ---
        self.is_animating = False
        for block in self.blocks:
            if block["animation_progress"] < 1.0:
                block["animation_progress"] += self.ANIMATION_SPEED
                block["animation_progress"] = min(1.0, block["animation_progress"])
                
                # Ease-out cubic interpolation
                t = block["animation_progress"]
                eased_t = 1 - pow(1 - t, 3)
                
                block["visual_pos"][0] = block["start_pos"][0] + (block["pos"][0] - block["start_pos"][0]) * eased_t
                block["visual_pos"][1] = block["start_pos"][1] + (block["pos"][1] - block["start_pos"][1]) * eased_t
                self.is_animating = True

        # Update selector animation
        sx, sy = self.selector_pos
        self.selector_visual_pos[0] += (sx - self.selector_visual_pos[0]) * self.ANIMATION_SPEED * 2
        self.selector_visual_pos[1] += (sy - self.selector_visual_pos[1]) * self.ANIMATION_SPEED * 2
        if abs(sx - self.selector_visual_pos[0]) > 0.01 or abs(sy - self.selector_visual_pos[1]) > 0.01:
            self.is_animating = True

        # --- Process Actions if not animating ---
        if not self.is_animating:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            is_push_action = space_held and movement != 0
            is_move_action = not space_held and movement != 0

            if is_push_action:
                reward += self._handle_push(movement)
            elif is_move_action:
                self._handle_selector_move(movement)

        # --- Update Game State & Rewards ---
        all_on_target = all(b['is_on_target'] for b in self.blocks)
        terminated = False
        
        if all_on_target:
            if not self.game_over: # First frame of winning
                reward += 100.0
                self.score += 100
                self.level += 1 # Progress to next level on reset
                # Spawn win particles
                for b in self.blocks:
                    self._spawn_particles(b['pos'], b['color'], 30)
            terminated = True
            self.game_over = True

        elif self.moves_left <= 0:
            if not self.game_over: # First frame of losing
                reward -= 10.0
                self.score -= 10
            terminated = True
            self.game_over = True
        
        if self.steps >= 1000:
            terminated = True
            self.game_over = True

        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_selector_move(self, movement):
        dx, dy = self._get_delta(movement)
        x, y = self.selector_pos
        self.selector_pos = ((x + dx) % self.GRID_W, (y + dy) % self.GRID_H)

    def _handle_push(self, movement):
        block_to_push = None
        for block in self.blocks:
            if block["pos"] == self.selector_pos and not block["is_on_target"]:
                block_to_push = block
                break
        
        if not block_to_push:
            return 0.0 # No pushable block under selector

        self.moves_left -= 1
        dx, dy = self._get_delta(movement)

        # Calculate path
        current_pos = list(block_to_push["pos"])
        path = [tuple(current_pos)]
        while True:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            # Check wall collision
            if not (0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H):
                break
            # Check block collision
            if any(b["pos"] == next_pos for b in self.blocks):
                break
            current_pos[0] += dx
            current_pos[1] += dy
            path.append(tuple(current_pos))
        
        final_pos = path[-1]
        
        if final_pos == block_to_push["pos"]: # Block couldn't move
            # Add bump effect
            self._spawn_particles(block_to_push["pos"], (150, 150, 150), 5, direction=(dx, dy))
            return -0.01 # Small penalty for failed push
        
        # Calculate reward for distance change
        old_dist = self._manhattan_distance(block_to_push["pos"], block_to_push["target_pos"])
        new_dist = self._manhattan_distance(final_pos, block_to_push["target_pos"])
        reward = (old_dist - new_dist) * 0.1

        # Update block state
        block_to_push["start_pos"] = list(block_to_push["visual_pos"])
        block_to_push["pos"] = final_pos
        block_to_push["animation_progress"] = 0.0
        
        # Check for landing on target
        if final_pos == block_to_push["target_pos"]:
            if not block_to_push["is_on_target"]:
                block_to_push["is_on_target"] = True
                reward += 1.0
                self.score += 1
                self._spawn_particles(final_pos, block_to_push["color"], 20)
                # Sound: Target acquired
        else:
            if block_to_push["is_on_target"]:
                block_to_push["is_on_target"] = False
                reward -= 1.0 # Penalty for moving off target
                self.score -= 1
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_W + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_H * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for y in range(self.GRID_H + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_W * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw targets
        for block in self.blocks:
            tx, ty = block["target_pos"]
            px = self.GRID_OFFSET_X + tx * self.CELL_SIZE
            py = self.GRID_OFFSET_Y + ty * self.CELL_SIZE
            rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect)
            pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, self.CELL_SIZE // 4, block["color"])
            pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, self.CELL_SIZE // 4, block["color"])

        # Draw blocks
        for block in self.blocks:
            vx, vy = block["visual_pos"]
            px = self.GRID_OFFSET_X + vx * self.CELL_SIZE
            py = self.GRID_OFFSET_Y + vy * self.CELL_SIZE
            rect = pygame.Rect(px + 4, py + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            
            shadow_color = tuple(c * 0.5 for c in block["color"])
            shadow_rect = rect.copy()
            shadow_rect.y += 2
            pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_radius=4)
            
            pygame.draw.rect(self.screen, block["color"], rect, border_radius=4)

            if block["is_on_target"]:
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=4)
                
        # Draw selector
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        alpha = 150 + int(pulse * 105)
        color = (*self.COLOR_SELECTOR, alpha)
        
        sel_x, sel_y = self.selector_visual_pos
        px = self.GRID_OFFSET_X + sel_x * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + sel_y * self.CELL_SIZE
        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        temp_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(temp_surf, color, (0, 0, *rect.size), 4, border_radius=6)
        self.screen.blit(temp_surf, rect.topleft)

    def _render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = p['life'] / p['max_life']
                color = (*p['color'], int(255 * alpha))
                size = int(p['size'] * alpha)
                if size > 0:
                    px = self.GRID_OFFSET_X + p['pos'][0] * self.CELL_SIZE + self.CELL_SIZE/2
                    py = self.GRID_OFFSET_Y + p['pos'][1] * self.CELL_SIZE + self.CELL_SIZE/2
                    pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), size, color)

    def _render_ui(self):
        ui_panel = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill((0, 0, 0, 100))
        self.screen.blit(ui_panel, (0, 0))

        moves_text = self.font_ui.render(f"Moves: {self.moves_left}/{self.max_moves}", True, (255, 255, 255))
        self.screen.blit(moves_text, (10, 10))
        
        score_text = self.font_ui.render(f"Score: {self.score:.2f}", True, (255, 255, 255))
        score_rect = score_text.get_rect(right=self.WIDTH - 10, top=10)
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "PUZZLE SOLVED!" if all(b['is_on_target'] for b in self.blocks) else "OUT OF MOVES"
            color = (0, 255, 150) if all(b['is_on_target'] for b in self.blocks) else (255, 100, 100)
            
            end_text = self.font_big.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "level": self.level,
        }

    def _get_delta(self, movement):
        if movement == 1: return 0, -1  # Up
        if movement == 2: return 0, 1   # Down
        if movement == 3: return -1, 0  # Left
        if movement == 4: return 1, 0   # Right
        return 0, 0

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _check_initial_targets(self):
        for block in self.blocks:
            if block["pos"] == block["target_pos"]:
                block["is_on_target"] = True
    
    def _spawn_particles(self, grid_pos, color, count, direction=None):
        for _ in range(count):
            if direction:
                angle = math.atan2(direction[1], direction[0]) + math.pi + self.rng.uniform(-0.5, 0.5)
            else:
                angle = self.rng.uniform(0, 2 * math.pi)
                
            speed = self.rng.uniform(0.01, 0.05)
            self.particles.append({
                'pos': list(grid_pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.rng.integers(20, 40),
                'max_life': 40,
                'color': color,
                'size': self.rng.integers(2, 5)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Controls ---
    # Use arrow keys to move selector
    # Hold SPACE and press an arrow key to push a block
    
    action = [0, 0, 0] # [movement, space, shift]
    
    while not done:
        # Pygame event handling for manual play
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0]

        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Buttons
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        
        # If the main script creates a display, blit to it
        try:
            screen = pygame.display.get_surface()
            if screen is None:
                screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        except pygame.error: # Handle headless environments
            pass

        if done:
            print(f"Game Over. Final Info: {info}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
            
    env.close()