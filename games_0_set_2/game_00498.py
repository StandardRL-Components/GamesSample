
# Generated: 2025-08-27T13:50:04.253995
# Source Brief: brief_00498.md
# Brief Index: 498

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# A simple structure for particles
Particle = namedtuple("Particle", ["pos", "vel", "radius", "color", "lifespan"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the block. Press space to drop it."
    )

    game_description = (
        "Stack blocks as high as possible in this isometric arcade game. "
        "Reach a height of 20 to win, but be careful! An unstable tower will collapse."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    TARGET_HEIGHT = 20

    # Colors
    COLOR_BG = (25, 28, 44)
    COLOR_GROUND = (60, 65, 80)
    COLOR_GROUND_SIDE = (45, 50, 65)
    COLOR_TARGET_LINE = (255, 255, 255)
    BLOCK_COLORS = [
        ((227, 85, 65), (191, 62, 53)),  # Red
        ((85, 188, 107), (65, 158, 86)),  # Green
        ((85, 144, 227), (65, 115, 191)), # Blue
        ((240, 216, 113), (212, 186, 80)), # Yellow
    ]
    COLOR_GHOST = (255, 255, 255, 60)
    COLOR_TEXT = (240, 240, 240)

    # Isometric projection parameters
    ISO_X_AXIS = np.array([0.5, -0.25]) * 32
    ISO_Y_AXIS = np.array([-0.5, -0.25]) * 32
    ISO_Z_AXIS = np.array([0, 0.5]) * 32
    
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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.render_mode = render_mode
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.game_over_animation_timer = 0
        
        self.camera_offset = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.8])
        
        # Game state
        self.placed_blocks = []
        self.falling_block = None
        self.unstable_blocks = []
        self.particles = []

        # Create the ground block
        ground_block = {
            "pos": np.array([-5.0, -1.0, -5.0]),
            "size": np.array([10.0, 1.0, 10.0]),
            "color": (self.COLOR_GROUND, self.COLOR_GROUND_SIDE),
            "center": np.array([0.0, -0.5, 0.0])
        }
        self.placed_blocks.append(ground_block)
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_pressed, _ = action
        space_pressed = space_pressed == 1
        reward = 0
        terminated = False

        if self.game_over_animation_timer > 0:
            self.game_over_animation_timer -= 1
            self._update_particles()
            if self.game_over_animation_timer == 0:
                terminated = True
            return self._get_observation(), 0, terminated, False, self._get_info()

        if self.game_won:
             return self._get_observation(), 100, True, False, self._get_info()

        self.steps += 1
        
        # --- Action Handling ---
        if space_pressed:
            # Drop the block
            reward = self._place_block()
            if self.game_over:
                reward = -100.0
                self.game_over_animation_timer = 60 # Start fall animation
            elif self.game_won:
                reward = 100.0
                terminated = True
        else:
            # Move the block
            prev_pos = self.falling_block['pos'].copy()
            move_speed = 0.5
            if movement == 1: # Up (Away from camera)
                self.falling_block['pos'][2] -= move_speed
            elif movement == 2: # Down (Towards camera)
                self.falling_block['pos'][2] += move_speed
            elif movement == 3: # Left
                self.falling_block['pos'][0] -= move_speed
            elif movement == 4: # Right
                self.falling_block['pos'][0] += move_speed
            
            # Movement reward/penalty
            if movement != 0:
                top_block = self.placed_blocks[-1]
                center_offset = np.linalg.norm(
                    (self.falling_block['pos'][[0,2]] + self.falling_block['size'][[0,2]] / 2) - 
                    (top_block['pos'][[0,2]] + top_block['size'][[0,2]] / 2)
                )
                reward = -0.05 * center_offset # Small penalty for being off-center

        # Clamp block position to prevent it from going too far
        self._clamp_falling_block()
        self._update_particles()
        
        if self.steps >= self.MAX_STEPS and not self.game_over and not self.game_won:
            terminated = True
            self.game_over = True
            reward = -10.0 # Penalty for running out of time

        return self._get_observation(), float(reward), terminated, False, self._get_info()

    def _place_block(self):
        # Add the falling block to the placed blocks
        placed_pos = self.falling_block['pos'].copy()
        placed_size = self.falling_block['size'].copy()
        new_block = {
            "pos": placed_pos,
            "size": placed_size,
            "color": self.falling_block['color'],
            "center": placed_pos + placed_size / 2
        }
        self.placed_blocks.append(new_block)
        # Sfx: Block place thud

        # Check stability
        if not self._check_stability():
            self.game_over = True
            # Sfx: Tower crumble
            return -100.0 # This reward will be returned in the main step loop

        # Calculate reward
        height = len(self.placed_blocks) - 1
        reward = 0.1 # Base reward for placing a block
        if height > 10:
            reward += 1.0 # Bonus for high placements

        # Create landing particles
        self._create_landing_particles(new_block)

        # Check for win condition
        if height >= self.TARGET_HEIGHT:
            self.game_won = True
            self.score += 100 # Add final bonus to score
            return 100.0

        # Spawn next block
        self.score += int(reward * 10)
        self._spawn_new_block()
        
        return reward

    def _check_stability(self):
        # Iterate from the second-to-top block down to the base
        for i in range(len(self.placed_blocks) - 2, -1, -1):
            support_block = self.placed_blocks[i]
            blocks_above = self.placed_blocks[i+1:]
            
            if not blocks_above: continue

            # Calculate center of mass of blocks_above
            com_x = sum(b['center'][0] for b in blocks_above) / len(blocks_above)
            com_z = sum(b['center'][2] for b in blocks_above) / len(blocks_above)

            # Check if CoM is within the support_block's footprint
            support_min_x = support_block['pos'][0]
            support_max_x = support_block['pos'][0] + support_block['size'][0]
            support_min_z = support_block['pos'][2]
            support_max_z = support_block['pos'][2] + support_block['size'][2]

            if not (support_min_x <= com_x <= support_max_x and support_min_z <= com_z <= support_max_z):
                self.unstable_blocks = blocks_above
                for block in self.unstable_blocks:
                    # Give them a random velocity for the fall animation
                    block['vel'] = np.array([
                        (self.np_random.random() - 0.5) * 0.2,
                        0.1, # Initial downward pop
                        (self.np_random.random() - 0.5) * 0.2
                    ])
                return False
        return True

    def _spawn_new_block(self):
        height = len(self.placed_blocks) - 1
        block_size = np.array([3.0, 1.0, 3.0])
        
        # New block starts centered above the previous one
        top_block = self.placed_blocks[-1]
        start_pos_xz = top_block['center'][[0,2]] - block_size[[0,2]] / 2
        
        self.falling_block = {
            "pos": np.array([start_pos_xz[0], float(height), start_pos_xz[1]]),
            "size": block_size,
            "color": self.BLOCK_COLORS[height % len(self.BLOCK_COLORS)],
        }
        
        # Adjust camera
        target_cam_y = self.SCREEN_HEIGHT * 0.8 - height * self.ISO_Z_AXIS[1] * self.falling_block['size'][1]
        self.camera_offset[1] += (target_cam_y - self.camera_offset[1]) * 0.5

    def _clamp_falling_block(self):
        top_block = self.placed_blocks[-1]
        limit = (top_block['size'][0] / 2) + (self.falling_block['size'][0] / 2)
        
        center_x = top_block['center'][0]
        self.falling_block['pos'][0] = np.clip(
            self.falling_block['pos'][0], center_x - limit, center_x + limit - self.falling_block['size'][0]
        )
        
        center_z = top_block['center'][2]
        self.falling_block['pos'][2] = np.clip(
            self.falling_block['pos'][2], center_z - limit, center_z + limit - self.falling_block['size'][2]
        )

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
            "height": max(0, len(self.placed_blocks) - 1),
        }

    # --- Rendering ---
    def _project(self, pos):
        return pos[0] * self.ISO_X_AXIS + pos[1] * self.ISO_Z_AXIS + pos[2] * self.ISO_Y_AXIS + self.camera_offset

    def _draw_cube(self, surface, block, color_override=None):
        pos, size = block['pos'], block['size']
        main_color, side_color = color_override if color_override else block['color']

        # Cube vertices in 3D space
        points = [
            pos + np.array([0, 0, 0]),
            pos + np.array([size[0], 0, 0]),
            pos + np.array([size[0], size[2], 0]),
            pos + np.array([0, size[2], 0]),
            pos + np.array([0, 0, size[1]]),
            pos + np.array([size[0], 0, size[1]]),
            pos + np.array([size[0], size[2], size[1]]),
            pos + np.array([0, size[2], size[1]]),
        ]
        
        # Project to 2D screen space
        screen_points = [self._project(p).astype(int) for p in points]
        
        # Draw faces
        # Right face
        pygame.gfxdraw.filled_polygon(surface, [screen_points[1], screen_points[2], screen_points[6], screen_points[5]], side_color)
        # Left face
        pygame.gfxdraw.filled_polygon(surface, [screen_points[0], screen_points[3], screen_points[7], screen_points[4]], side_color)
        # Top face
        pygame.gfxdraw.filled_polygon(surface, [screen_points[4], screen_points[5], screen_points[6], screen_points[7]], main_color)
        # Outlines for clarity
        pygame.gfxdraw.aapolygon(surface, [screen_points[1], screen_points[2], screen_points[6], screen_points[5]], side_color)
        pygame.gfxdraw.aapolygon(surface, [screen_points[0], screen_points[3], screen_points[7], screen_points[4]], side_color)
        pygame.gfxdraw.aapolygon(surface, [screen_points[4], screen_points[5], screen_points[6], screen_points[7]], main_color)


    def _render_game(self):
        # Sort all blocks by a heuristic y-depth for correct Z-ordering
        render_queue = self.placed_blocks[:]
        if self.falling_block and not self.game_over:
             # Draw ghost block
            ghost_block = self.falling_block.copy()
            self._draw_cube(self.screen, ghost_block, color_override=(self.COLOR_GHOST, self.COLOR_GHOST))
            render_queue.append(self.falling_block)

        # Animate falling blocks on game over
        if self.game_over:
            for block in self.unstable_blocks:
                block['vel'][1] += 0.01 # Gravity
                block['pos'] += block['vel']
            render_queue.extend(self.unstable_blocks)

        # Sort by depth for correct rendering order
        render_queue.sort(key=lambda b: b['pos'][0] + b['pos'][2] + b['pos'][1])

        for block in render_queue:
            self._draw_cube(self.screen, block)
        
        self._render_particles()

        # Draw target height line
        p1 = self._project(np.array([-6, self.TARGET_HEIGHT, -6]))
        p2 = self._project(np.array([6, self.TARGET_HEIGHT, -6]))
        p3 = self._project(np.array([6, self.TARGET_HEIGHT, 6]))
        pygame.draw.aaline(self.screen, self.COLOR_TARGET_LINE, p1, p2, 1)
        pygame.draw.aaline(self.screen, self.COLOR_TARGET_LINE, p2, p3, 1)

    def _render_ui(self):
        height = max(0, len(self.placed_blocks) - 1)
        
        # Height display
        height_text = self.font_large.render(f"{height}", True, self.COLOR_TEXT)
        height_label = self.font_small.render("HEIGHT", True, self.COLOR_TEXT)
        self.screen.blit(height_label, (20, 15))
        self.screen.blit(height_text, (20, 40))

        # Score display
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        score_label = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_label, (self.SCREEN_WIDTH - score_label.get_width() - 20, 15))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 40))

        if self.game_won:
            win_text = self.font_large.render("TOWER COMPLETE!", True, self.BLOCK_COLORS[3][0])
            self.screen.blit(win_text, (self.SCREEN_WIDTH/2 - win_text.get_width()/2, self.SCREEN_HEIGHT/2 - win_text.get_height()/2))
        elif self.game_over and self.game_over_animation_timer > 0:
            lose_text = self.font_large.render("TOWER COLLAPSED!", True, self.BLOCK_COLORS[0][0])
            self.screen.blit(lose_text, (self.SCREEN_WIDTH/2 - lose_text.get_width()/2, self.SCREEN_HEIGHT/2 - lose_text.get_height()/2))

    # --- Particle System ---
    def _create_landing_particles(self, block):
        center_pos = block['pos'] + block['size'] / 2
        for _ in range(15):
            vel = np.array([
                (self.np_random.random() - 0.5) * 0.4,
                -self.np_random.random() * 0.1,
                (self.np_random.random() - 0.5) * 0.4,
            ])
            self.particles.append(Particle(
                pos=center_pos.copy(),
                vel=vel,
                radius=self.np_random.random() * 3 + 2,
                color=block['color'][0],
                lifespan=self.np_random.integers(15, 30)
            ))

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            new_pos = p.pos + p.vel
            new_radius = p.radius * 0.95
            new_lifespan = p.lifespan - 1
            if new_lifespan > 0 and new_radius > 0.5:
                new_particles.append(p._replace(pos=new_pos, radius=new_radius, lifespan=new_lifespan))
        self.particles = new_particles

    def _render_particles(self):
        for p in self.particles:
            screen_pos = self._project(p.pos)
            pygame.gfxdraw.filled_circle(
                self.screen, int(screen_pos[0]), int(screen_pos[1]), int(p.radius), p.color
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(screen_pos[0]), int(screen_pos[1]), int(p.radius), p.color
            )
            
    def close(self):
        pygame.quit()

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

# Example of how to run the environment
if __name__ == '__main__':
    # This part allows for human play
    import sys
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Isometric Stacker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("--- Human Player Mode ---")
    print(env.user_guide)
    print("-------------------------")

    while not terminated:
        # Default action is no-op
        action = [0, 0, 0] # move=none, space=released, shift=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        if keys[pygame.K_SPACE]:
            action[1] = 1

        if keys[pygame.K_r]: # Allow resetting mid-game
            obs, info = env.reset()
            total_reward = 0
            print("--- Game Reset ---")
            continue

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term or trunc
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we need a small delay for human playability
        clock.tick(30)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before closing or resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            terminated = False # Restart for continuous play
            total_reward = 0

    env.close()
    pygame.quit()
    sys.exit()