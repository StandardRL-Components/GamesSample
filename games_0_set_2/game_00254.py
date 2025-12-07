
# Generated: 2025-08-27T13:05:02.480309
# Source Brief: brief_00254.md
# Brief Index: 254

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric block-stacking game where the goal is to build a tower as high as
    possible without it collapsing. Players control the horizontal position of a
    falling block and can drop it into place. The tower's stability is simulated,
    and unstable placements will cause the tower to wobble and eventually fall,
    ending the game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to move the falling block. "
        "Hold Space to drop it faster. Stack 20 blocks to win."
    )

    game_description = (
        "Stack colorful, isometric blocks as high as possible before the tower "
        "collapses in this challenging puzzle game."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GROUND_LEVEL_Z = 50  # Screen Y-pos of the ground
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (30, 40, 50)
    COLOR_GROUND = (40, 50, 60)
    COLOR_GROUND_SIDE = (35, 45, 55)
    COLOR_TEXT = (220, 220, 230)
    COLOR_SCORE = (255, 200, 0)
    
    # Block types: [color, darker_side, weight]
    BLOCK_TYPES = [
        ((220, 50, 50), (180, 40, 40), 1.5),   # Red (heavy)
        ((50, 220, 50), (40, 180, 40), 1.0),   # Green
        ((50, 150, 220), (40, 120, 180), 1.0), # Blue
        ((220, 200, 50), (180, 160, 40), 1.0), # Yellow
        ((180, 50, 220), (140, 40, 180), 1.0), # Purple
    ]

    # Isometric projection constants
    BLOCK_WIDTH_ISO = 48
    BLOCK_HEIGHT_ISO = 24
    BLOCK_DEPTH_VISUAL = 20

    # Game mechanics
    MAX_STEPS = 1500
    WIN_HEIGHT = 20
    FALL_SPEED = 0.5
    FAST_FALL_SPEED = 2.0
    MOVE_SPEED = 1.5
    GRID_WIDTH = 10 # Number of possible block positions horizontally
    
    # Physics
    WOBBLE_PROPAGATION = 0.6
    WOBBLE_DAMPING = 0.95
    PLACEMENT_WOBBLE_AMOUNT = 0.1
    COLLAPSE_THRESHOLD = 1.5 # Radians

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        self.font_huge = pygame.font.Font(None, 72)

        self.render_mode = render_mode
        self.np_random = None

        # These will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stacked_blocks = []
        self.current_block = None
        self.next_block_type_idx = 0
        self.particles = []
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stacked_blocks = []
        self.current_block = None
        self.next_block_type_idx = self.np_random.integers(0, len(self.BLOCK_TYPES))
        self.particles = []
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0
        terminated = False
        
        self.steps += 1
        
        if not self.game_over:
            self._update_current_block(movement, space_held)
            self._update_physics()
            
            # Check for landing
            collision_info = self._check_collision()
            if collision_info['collided']:
                placement_reward = self._place_block(collision_info)
                reward += placement_reward
                
                self._update_stability()
                self._check_for_collapse()
                
                if not self.game_over:
                    if len(self.stacked_blocks) >= self.WIN_HEIGHT:
                        self.win = True
                        self.game_over = True
                        reward += 100 # Win reward
                    else:
                        self._spawn_new_block()

        self.score += reward
        
        if self.game_over or self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.win:
                 # Ensure collapse reward is only given once
                if "collapse_reward_given" not in self._get_info():
                    reward -= 10 # Collapse penalty
                    self.score -= 10
                    self.info["collapse_reward_given"] = True


        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_current_block(self, movement, space_held):
        if self.current_block is None:
            return

        # Horizontal movement
        if movement == 3: # Left
            self.current_block['pos'][0] -= self.MOVE_SPEED
        elif movement == 4: # Right
            self.current_block['pos'][0] += self.MOVE_SPEED
        
        # Clamp position
        max_x = self.GRID_WIDTH - 1
        self.current_block['pos'][0] = np.clip(self.current_block['pos'][0], 0, max_x)

        # Vertical movement (gravity)
        fall_speed = self.FAST_FALL_SPEED if space_held else self.FALL_SPEED
        self.current_block['pos'][1] -= fall_speed

    def _check_collision(self):
        if self.current_block is None:
            return {'collided': False}

        block_x, block_z = self.current_block['pos']
        
        # Ground collision
        if block_z <= 0:
            return {'collided': True, 'support_y': 0, 'support_center_x': block_x, 'support_width': self.GRID_WIDTH}
            
        # Stack collision
        top_z = -1
        support_blocks = []
        for sb in self.stacked_blocks:
            if abs(sb['pos'][0] - block_x) < 1.0: # Check for horizontal overlap
                if sb['pos'][1] > top_z:
                    top_z = sb['pos'][1]
                    support_blocks = [sb]
                elif sb['pos'][1] == top_z:
                    support_blocks.append(sb)

        if block_z <= top_z + 1:
            if not support_blocks:
                return {'collided': False} # Fell through a gap
            
            support_xs = [sb['pos'][0] for sb in support_blocks]
            support_center_x = sum(support_xs) / len(support_xs)
            support_width = (max(support_xs) - min(support_xs)) + 1
            return {'collided': True, 'support_y': top_z + 1, 'support_center_x': support_center_x, 'support_width': support_width}

        return {'collided': False}

    def _place_block(self, collision_info):
        # Snap to grid
        self.current_block['pos'][0] = round(self.current_block['pos'][0])
        self.current_block['pos'][1] = collision_info['support_y']
        
        # Add to stack
        self.stacked_blocks.append(self.current_block)
        
        # Add placement particles
        iso_pos = self._project_iso(self.current_block['pos'][0], self.current_block['pos'][1])
        self._create_particles(iso_pos[0], iso_pos[1] + self.BLOCK_HEIGHT_ISO, self.current_block['color'])
        
        # Calculate reward
        overhang = abs(self.current_block['pos'][0] - collision_info['support_center_x'])
        
        # Add wobble based on placement
        wobble_force = overhang * self.PLACEMENT_WOBBLE_AMOUNT
        self.stacked_blocks[-1]['wobble_vel'] += wobble_force * (1 if self.current_block['pos'][0] > collision_info['support_center_x'] else -1)

        self.current_block = None
        
        if overhang > 0.5 * collision_info['support_width']:
            return 1.0 # Risky placement reward
        elif overhang < 0.1:
            return -0.02 # Safe placement penalty
        else:
            return 0.1 # Standard placement reward

    def _spawn_new_block(self):
        block_type_idx = self.next_block_type_idx
        color, dark_color, weight = self.BLOCK_TYPES[block_type_idx]
        
        self.current_block = {
            'pos': [self.GRID_WIDTH / 2, self.WIN_HEIGHT + 2], # x, z
            'type_idx': block_type_idx,
            'color': color,
            'dark_color': dark_color,
            'weight': weight,
            'wobble': 0.0,
            'wobble_vel': 0.0,
        }
        self.next_block_type_idx = self.np_random.integers(0, len(self.BLOCK_TYPES))
        
    def _update_physics(self):
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2 # Gravity
            p['life'] -= 1

    def _update_stability(self):
        # Propagate wobble up the stack
        for i in range(len(self.stacked_blocks) - 2, -1, -1):
            block_above = self.stacked_blocks[i+1]
            block_current = self.stacked_blocks[i]
            
            # Transfer some wobble velocity
            transfer_vel = block_above['wobble_vel'] * self.WOBBLE_PROPAGATION
            block_current['wobble_vel'] += transfer_vel
            block_above['wobble_vel'] -= transfer_vel # Action-reaction

        # Update all block wobbles
        for block in self.stacked_blocks:
            # Apply wobble velocity to angle, and dampen
            block['wobble'] += block['wobble_vel']
            block['wobble_vel'] *= self.WOBBLE_DAMPING
            
            # Restoring force (center of mass pulling it back)
            # A simple approximation: pull back towards zero
            block['wobble_vel'] -= block['wobble'] * 0.01 * block['weight']

    def _check_for_collapse(self):
        for block in self.stacked_blocks:
            if abs(block['wobble']) > self.COLLAPSE_THRESHOLD:
                self.game_over = True
                # Create a large particle explosion
                for b in self.stacked_blocks:
                    iso_pos = self._project_iso(b['pos'][0], b['pos'][1])
                    self._create_particles(iso_pos[0], iso_pos[1], b['color'], count=20, power=5)
                # Sound: sfx_tower_collapse.wav
                return

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        info = {
            "score": self.score,
            "steps": self.steps,
            "height": len(self.stacked_blocks),
        }
        if self.game_over and not self.win:
            info["collapse_reward_given"] = True
        return info

    def _project_iso(self, x, z):
        """Converts 2D grid coordinates (x, z) to 2D screen coordinates."""
        screen_x = self.SCREEN_WIDTH // 2 + (x - self.GRID_WIDTH / 2) * self.BLOCK_WIDTH_ISO / 2
        screen_y = self.GROUND_LEVEL_Z + self.WIN_HEIGHT * self.BLOCK_DEPTH_VISUAL - z * self.BLOCK_DEPTH_VISUAL
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, x, z, color, dark_color, wobble=0.0):
        """Draws a single isometric cube with wobble."""
        sx, sy = self._project_iso(x, z)
        
        w_half, h_half = self.BLOCK_WIDTH_ISO / 2, self.BLOCK_HEIGHT_ISO / 2
        
        points = [
            (0, -h_half), (-w_half, 0), (0, h_half), (w_half, 0) # Top face
        ]
        
        # Apply wobble rotation
        if wobble != 0.0:
            cos_w, sin_w = math.cos(wobble), math.sin(wobble)
            points = [(p[0] * cos_w - p[1] * sin_w, p[0] * sin_w + p[1] * cos_w) for p in points]

        # Top face
        top_points = [(sx + p[0], sy + p[1]) for p in points]
        
        # Side faces
        left_side = [top_points[0], top_points[1], (top_points[1][0], top_points[1][1] + self.BLOCK_DEPTH_VISUAL), (top_points[0][0], top_points[0][1] + self.BLOCK_DEPTH_VISUAL)]
        right_side = [top_points[0], top_points[3], (top_points[3][0], top_points[3][1] + self.BLOCK_DEPTH_VISUAL), (top_points[0][0], top_points[0][1] + self.BLOCK_DEPTH_VISUAL)]

        # Draw with antialiasing
        pygame.gfxdraw.filled_polygon(surface, left_side, dark_color)
        pygame.gfxdraw.aapolygon(surface, left_side, dark_color)
        pygame.gfxdraw.filled_polygon(surface, right_side, dark_color)
        pygame.gfxdraw.aapolygon(surface, right_side, dark_color)
        pygame.gfxdraw.filled_polygon(surface, top_points, color)
        pygame.gfxdraw.aapolygon(surface, top_points, color)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_WIDTH + 1):
            start = self._project_iso(i, 0)
            end = self._project_iso(i, self.WIN_HEIGHT + 5)
            pygame.draw.line(self.screen, self.COLOR_GRID, (start[0], self.GROUND_LEVEL_Z), end, 1)
        
        # Draw ground
        ground_points = [
            self._project_iso(0, 0),
            self._project_iso(self.GRID_WIDTH, 0),
            (self._project_iso(self.GRID_WIDTH, 0)[0], self.SCREEN_HEIGHT),
            (self._project_iso(0, 0)[0], self.SCREEN_HEIGHT)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, ground_points, self.COLOR_GROUND_SIDE)

        # Draw stacked blocks
        for block in sorted(self.stacked_blocks, key=lambda b: b['pos'][1]):
            self._draw_iso_cube(self.screen, block['pos'][0], block['pos'][1], block['color'], block['dark_color'], block['wobble'])

        # Draw current block's shadow
        if self.current_block:
            collision = self._check_collision()
            shadow_z = 0
            if not collision['collided']:
                # Find where it would land
                top_z = -1
                for sb in self.stacked_blocks:
                     if abs(sb['pos'][0] - self.current_block['pos'][0]) < 1.0:
                         top_z = max(top_z, sb['pos'][1])
                shadow_z = top_z + 1

            sx, sy = self._project_iso(self.current_block['pos'][0], shadow_z)
            shadow_rect = pygame.Rect(0, 0, self.BLOCK_WIDTH_ISO, self.BLOCK_HEIGHT_ISO)
            shadow_rect.center = (sx, sy + self.BLOCK_HEIGHT_ISO/2 + self.BLOCK_DEPTH_VISUAL)
            shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, (0, 0, 0, 80), (0, 0, shadow_rect.width, shadow_rect.height))
            self.screen.blit(shadow_surf, shadow_rect.topleft)

        # Draw current block
        if self.current_block and not self.game_over:
            self._draw_iso_cube(self.screen, self.current_block['pos'][0], self.current_block['pos'][1], self.current_block['color'], self.current_block['dark_color'])

        # Draw particles
        for p in self.particles:
            size = max(1, int(p['life'] / p['max_life'] * 5))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        # Score and Height
        score_text = self.font_large.render(f"{self.score:.1f}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 10))
        
        height_text = self.font_large.render(f"{len(self.stacked_blocks)} / {self.WIN_HEIGHT}", True, self.COLOR_TEXT)
        height_rect = height_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(height_text, height_rect)
        
        # Next Block Preview
        next_block_color, next_block_dark, _ = self.BLOCK_TYPES[self.next_block_type_idx]
        self._draw_iso_cube(self.screen, -2.5, self.WIN_HEIGHT, next_block_color, next_block_dark)

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text = self.font_huge.render("YOU WIN!", True, (100, 255, 100))
            else:
                end_text = self.font_huge.render("TOWER COLLAPSED", True, (255, 100, 100))
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, x, y, color, count=10, power=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, power)
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 1],
                'life': life,
                'max_life': life,
                'color': color
            })
    
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Isometric Tower Builder")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame specific event handling and rendering ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Transpose the observation for Pygame's display format (W, H, C)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}. Height: {info['height']}")
            # Wait for 'R' to reset
            wait_for_reset = True
            while wait_for_reset:
                 for event in pygame.event.get():
                     if event.type == pygame.QUIT:
                         wait_for_reset = False
                         running = False
                     if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                         obs, info = env.reset()
                         total_reward = 0
                         wait_for_reset = False
        
        clock.tick(30) # Run at 30 FPS

    env.close()