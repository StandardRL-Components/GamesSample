import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:26:10.141835
# Source Brief: brief_00975.md
# Brief Index: 975
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Block:
    """A simple class to hold the state of a single block in the physics simulation."""
    def __init__(self, pos, size, color):
        self.pos = np.array(pos, dtype=float)
        self.size = np.array(size, dtype=float)
        self.vel = np.array([0.0, 0.0, 0.0])
        self.color = color
        self.supported = False # True if resting on the ground or another block

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must build a stable tower of blocks.
    The goal is to reach a specific height before running out of blocks or the tower collapsing.
    This environment features a simplified 3D physics simulation and an isometric visual style.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Build a stable tower of blocks up to a target height. Place blocks strategically before they fall, and prevent the tower from collapsing under its own weight."
    user_guide = "Use the arrow keys (↑↓←→) to move the cursor on the ground. Press space to drop the current block."
    auto_advance = True
    
    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1000
    TOTAL_BLOCKS = 50
    WIN_HEIGHT = 100.0
    
    # Physics
    GRAVITY = 0.05
    FRICTION = 0.98
    BOUNCINESS = 0.1
    PHYSICS_SUBSTEPS = 4
    
    # Visuals & Controls
    CURSOR_SPEED = 1.5
    PLAY_AREA_RADIUS = 50

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GROUND = (40, 60, 80)
    COLOR_GROUND_GRID = (50, 70, 90)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_WIN_LINE = (0, 255, 0, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    BLOCK_COLORS = [
        (227, 88, 88), (98, 171, 232), (141, 224, 121), 
        (230, 163, 90), (188, 132, 232), (232, 224, 132)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Define the vectors for isometric projection
        self.iso_x_ax = np.array([0.5, 0.25]) * 20
        self.iso_y_ax = np.array([-0.5, 0.25]) * 20
        self.iso_z_ax = np.array([0, -1]) * 20
        
        # Initialize state variables
        self.blocks = []
        self.falling_block = None
        self.cursor_pos = np.array([0.0, 0.0]) # X and Z cursor pos
        self.particles = []
        self.last_space_held = False
        self.max_tower_height = 0.0
        self.blocks_placed_count = 0
        self.initial_fall_speed = 0.5
        
        # self.reset() # This was called before super().reset() in the original code, moved to be correct
        # self.validate_implementation() # This should not be here

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.blocks = []
        self.particles = []
        self.cursor_pos = np.array([0.0, 0.0])
        self.last_space_held = False
        self.max_tower_height = 0.0
        self.blocks_placed_count = 0
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for every step to encourage efficiency
        
        if not self.game_over:
            placement_reward = self._handle_input(action)
            reward += placement_reward

            self._update_physics()
            self._update_particles()
            
            new_height = self._calculate_tower_height()
            if new_height > self.max_tower_height:
                reward += 10.0
                self.max_tower_height = new_height
            
            fallen_blocks_count = self._cleanup_fallen_blocks()
            if fallen_blocks_count > 0:
                reward -= 5.0 * fallen_blocks_count
                # // SFX: Block falls off
            
            self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if (terminated or truncated) and not self.game_over:
            self.game_over = True
            if self.max_tower_height >= self.WIN_HEIGHT:
                reward += 100.0 # Win
            elif self.blocks_placed_count >= self.TOTAL_BLOCKS:
                reward -= 10.0 # Loss by running out of blocks
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_new_block(self):
        if self.blocks_placed_count >= self.TOTAL_BLOCKS:
            self.falling_block = None
            return

        size = np.array([10.0, 10.0, 10.0])
        x_spawn = self.np_random.uniform(-self.PLAY_AREA_RADIUS * 0.5, self.PLAY_AREA_RADIUS * 0.5)
        z_spawn = self.np_random.uniform(-self.PLAY_AREA_RADIUS * 0.5, self.PLAY_AREA_RADIUS * 0.5)
        
        self.falling_block = Block(
            pos=[x_spawn, self.WIN_HEIGHT + 30, z_spawn],
            size=size,
            color=random.choice(self.BLOCK_COLORS)
        )
        fall_speed_increase = 0.01 * (self.blocks_placed_count // 5)
        self.falling_block.vel[1] = -(self.initial_fall_speed + fall_speed_increase)

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement (map to XZ plane) ---
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED # Up -> Away (Z-)
        if movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED # Down -> Towards (Z+)
        if movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED # Left (X-)
        if movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED # Right (X+)
        
        self.cursor_pos = np.clip(self.cursor_pos, -self.PLAY_AREA_RADIUS, self.PLAY_AREA_RADIUS)
        
        # --- Place Block ---
        if space_held and not self.last_space_held and self.falling_block:
            # // SFX: Block placed
            self.falling_block.pos[0] = self.cursor_pos[0]
            self.falling_block.pos[2] = self.cursor_pos[1]
            self.blocks.append(self.falling_block)
            
            self.blocks_placed_count += 1
            self._spawn_new_block()
            self.last_space_held = space_held
            return 0.1 # Reward for placing a block
        
        self.last_space_held = space_held
        return 0.0

    def _update_physics(self):
        # Update free-falling block (the one player controls)
        if self.falling_block:
            self.falling_block.pos += self.falling_block.vel
            if self.falling_block.pos[1] < -50: # Fell off screen before placement
                self._spawn_new_block()

        # Simplified rigid body physics for placed blocks
        for _ in range(self.PHYSICS_SUBSTEPS):
            for block in self.blocks:
                block.supported = False
                block.vel[1] -= self.GRAVITY / self.PHYSICS_SUBSTEPS

            for i, block_a in enumerate(self.blocks):
                # Ground collision
                if block_a.pos[1] - block_a.size[1] / 2 < 0:
                    block_a.pos[1] = block_a.size[1] / 2
                    if block_a.vel[1] < 0:
                        block_a.vel[1] *= -self.BOUNCINESS
                    block_a.vel[0] *= self.FRICTION
                    block_a.vel[2] *= self.FRICTION
                    block_a.supported = True
                
                # Block-block collision
                for block_b in self.blocks[i+1:]:
                    self._collide(block_a, block_b)

            for block in self.blocks:
                block.pos += block.vel / self.PHYSICS_SUBSTEPS

    def _collide(self, a, b):
        delta = a.pos - b.pos
        half_size_a = a.size / 2
        half_size_b = b.size / 2
        
        overlap_x = (half_size_a[0] + half_size_b[0]) - abs(delta[0])
        overlap_y = (half_size_a[1] + half_size_b[1]) - abs(delta[1])
        overlap_z = (half_size_a[2] + half_size_b[2]) - abs(delta[2])

        if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
            min_overlap = min(overlap_x, overlap_y, overlap_z)
            
            if min_overlap == overlap_y:
                pen_vec = np.array([0, np.sign(delta[1]) * overlap_y, 0]) if delta[1] != 0 else np.array([0, overlap_y, 0])
                rel_vel_y = a.vel[1] - b.vel[1]
                impulse = -(1 + self.BOUNCINESS) * rel_vel_y / 2
                a.vel[1] += impulse
                b.vel[1] -= impulse
                
                friction_impulse_x = -np.clip((a.vel[0] - b.vel[0]), -0.5, 0.5) * self.FRICTION * 0.1
                friction_impulse_z = -np.clip((a.vel[2] - b.vel[2]), -0.5, 0.5) * self.FRICTION * 0.1
                a.vel[0] += friction_impulse_x; b.vel[0] -= friction_impulse_x
                a.vel[2] += friction_impulse_z; b.vel[2] -= friction_impulse_z

                a.supported = a.supported or (a.pos[1] > b.pos[1])
                b.supported = b.supported or (b.pos[1] > a.pos[1])
            elif min_overlap == overlap_x:
                pen_vec = np.array([np.sign(delta[0]) * overlap_x, 0, 0]) if delta[0] != 0 else np.array([overlap_x, 0, 0])
                rel_vel_x = a.vel[0] - b.vel[0]
                impulse = -(1 + self.BOUNCINESS) * rel_vel_x / 2
                a.vel[0] += impulse; b.vel[0] -= impulse
            else: # min_overlap == overlap_z
                pen_vec = np.array([0, 0, np.sign(delta[2]) * overlap_z]) if delta[2] != 0 else np.array([0, 0, overlap_z])
                rel_vel_z = a.vel[2] - b.vel[2]
                impulse = -(1 + self.BOUNCINESS) * rel_vel_z / 2
                a.vel[2] += impulse; b.vel[2] -= impulse
            
            a.pos += pen_vec / 2; b.pos -= pen_vec / 2

            if np.linalg.norm(a.vel - b.vel) > 0.1 and len(self.particles) < 100:
                for _ in range(3):
                    self.particles.append({
                        'pos': (a.pos + b.pos) / 2,
                        'vel': self.np_random.standard_normal(3) * 0.5,
                        'life': 20,
                        'color': (200, 200, 200)
                    })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] -= self.GRAVITY * 0.5
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _calculate_tower_height(self):
        if not self.blocks: return 0.0
        max_h = 0.0
        for block in self.blocks:
            if block.supported:
                h = block.pos[1] + block.size[1] / 2
                if h > max_h: max_h = h
        return max_h

    def _cleanup_fallen_blocks(self):
        initial_count = len(self.blocks)
        self.blocks = [b for b in self.blocks if b.pos[1] > -20]
        return initial_count - len(self.blocks)

    def _check_termination(self):
        if self.max_tower_height >= self.WIN_HEIGHT: return True
        if self.blocks_placed_count >= self.TOTAL_BLOCKS and self.falling_block is None: return True
        if self.blocks_placed_count >= 1 and not self.blocks: return True
        return False
        
    def _project(self, pos):
        screen_pos = np.array([self.WIDTH / 2, self.HEIGHT / 1.5])
        screen_pos += pos[0] * self.iso_x_ax
        screen_pos += pos[2] * self.iso_y_ax
        screen_pos += pos[1] * self.iso_z_ax
        return screen_pos.astype(int)

    def _draw_iso_box(self, surface, pos, size, color):
        half_size = size / 2
        corners_3d = [ pos + np.array([sx, sy, sz]) * half_size for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1] ]
        corners_2d = [self._project(c) for c in corners_3d]
        
        faces = [(4, 5, 7, 6), (1, 3, 7, 5), (2, 3, 7, 6)]
        top_color = tuple(min(255, int(c * 1.0)) for c in color)
        right_color = tuple(min(255, int(c * 0.8)) for c in color)
        left_color = tuple(min(255, int(c * 0.6)) for c in color)
        face_colors = [top_color, right_color, left_color]
        
        for face_indices, face_color in zip(faces, face_colors):
            points = [corners_2d[i] for i in face_indices]
            pygame.gfxdraw.filled_polygon(surface, points, face_color)
            pygame.gfxdraw.aapolygon(surface, points, (0,0,0,50))
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        r = self.PLAY_AREA_RADIUS + 10
        ground_points = [self._project(np.array([x, 0, z])) for x, z in [(r,r), (r,-r), (-r,-r), (-r,r)]]
        pygame.gfxdraw.filled_polygon(self.screen, ground_points, self.COLOR_GROUND)
        
        for i in np.linspace(-r, r, 11):
            pygame.draw.aaline(self.screen, self.COLOR_GROUND_GRID, self._project(np.array([i,0,-r])), self._project(np.array([i,0,r])))
            pygame.draw.aaline(self.screen, self.COLOR_GROUND_GRID, self._project(np.array([-r,0,i])), self._project(np.array([r,0,i])))
        
        draw_list = list(self.blocks)
        if self.falling_block and self.falling_block not in self.blocks: draw_list.append(self.falling_block)
        draw_list.sort(key=lambda b: b.pos[0] + b.pos[2] + b.pos[1], reverse=True)
        
        cursor_world_pos = np.array([self.cursor_pos[0], 0, self.cursor_pos[1]])
        self._draw_iso_box(self.screen, cursor_world_pos, np.array([10,0.5,10]), self.COLOR_CURSOR)
        
        win_line_l = self._project(np.array([-r, self.WIN_HEIGHT, -r]))
        win_line_r = self._project(np.array([r, self.WIN_HEIGHT, -r]))
        pygame.draw.line(self.screen, self.COLOR_WIN_LINE, win_line_l, win_line_r, 2)
        
        for block in draw_list:
            self._draw_iso_box(self.screen, block.pos, block.size, block.color)
            
        for p in self.particles:
            p_screen = self._project(p['pos'])
            pygame.draw.circle(self.screen, p['color'], p_screen, max(1, int(p['life']/5)))

    def _render_ui(self):
        def draw_text(text, font, pos, color=self.COLOR_TEXT, shadow_color=self.COLOR_TEXT_SHADOW):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        height_text = f"Height: {self.max_tower_height:.1f} / {self.WIN_HEIGHT}"
        blocks_text = f"Blocks: {self.TOTAL_BLOCKS - self.blocks_placed_count}"
        draw_text(height_text, self.font_small, (10, 10))
        draw_text(blocks_text, self.font_small, (self.WIDTH - 150, 10))
        
        if self.game_over:
            msg, color = ("VICTORY!", (100, 255, 100)) if self.max_tower_height >= self.WIN_HEIGHT else ("TOWER FAILED", (255, 100, 100))
            draw_text(msg, self.font_large, (self.WIDTH/2 - self.font_large.size(msg)[0]/2, self.HEIGHT/3), color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tower_height": self.max_tower_height,
            "blocks_remaining": self.TOTAL_BLOCKS - self.blocks_placed_count
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        # Test game-specific assertions
        assert 0 <= self._calculate_tower_height()
        assert 0 <= self.blocks_placed_count <= self.TOTAL_BLOCKS
        assert self.cursor_pos[0] >= -self.PLAY_AREA_RADIUS and self.cursor_pos[0] <= self.PLAY_AREA_RADIUS
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # The validation call was removed from __init__ to avoid issues with environment checkers
    # that may not expect it. It's better practice to call it externally if needed.
    env_to_validate = GameEnv()
    env_to_validate.validate_implementation()
    env_to_validate.close()

    env = GameEnv()
    obs, info = env.reset()
    
    # The main loop now requires a display to be created for human play
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Builder - Manual Control")
    
    key_to_action = { pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4 }
    total_reward = 0
    running = True

    while running:
        movement_action, space_action = 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- RESET ---")

        keys = pygame.key.get_pressed()
        for key, move in key_to_action.items():
            if keys[key]: movement_action = move
        if keys[pygame.K_SPACE]: space_action = 1
            
        action = [movement_action, space_action, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            print("Press 'r' to restart or 'q' to quit.")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()