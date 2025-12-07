import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:48:08.619358
# Source Brief: brief_00646.md
# Brief Index: 646
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro-style arcade racing game with puzzle elements.
    The player races through a procedurally generated track, matching their vehicle's
    color with tiles on the ground to trigger chain reactions for speed boosts and points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Race down a futuristic track, matching your ship's color with tiles on the road to trigger score combos and speed boosts."
    )
    user_guide = "Controls: Use ← and → arrow keys to steer your ship."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_UI_TEXT = (230, 230, 255)
    COLOR_PLAYER = (0, 255, 255) # Bright Cyan
    COLOR_PLAYER_GLOW = (0, 255, 255, 50)
    COLOR_WALL = (60, 60, 80)
    COLOR_FINISH_LINE = (255, 255, 255)
    TILE_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (50, 100, 255),  # Blue
        (255, 255, 50),  # Yellow
    ]

    # Game parameters
    MAX_STEPS = 5000
    TRACK_LENGTH = 10000 # World units
    TRACK_SEGMENT_LENGTH = 50
    BASE_SPEED = 4.0
    PLAYER_Y_POS = SCREEN_HEIGHT * 0.8
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = 0.90
    PLAYER_MAX_SPEED = 10.0
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.world_y = 0.0
        self.game_speed = self.BASE_SPEED
        
        # Player state
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_color_idx = 0

        # Game world elements
        self.track = []
        self.tiles = []
        self.particles = []
        
        # The original code called reset() and validate_implementation() here.
        # It's better practice to let the user call reset() explicitly.
        # self.reset()
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.world_y = 0.0
        self.game_speed = self.BASE_SPEED

        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.PLAYER_Y_POS])
        self.player_vel = np.array([0.0, 0.0])
        self.player_color_idx = self.np_random.integers(0, len(self.TILE_COLORS))

        self.particles = []
        self._generate_track()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        
        # --- Update Game Logic ---
        self.steps += 1
        
        # 1. Handle player input
        self._handle_input(movement)
        
        # 2. Update player physics
        self._update_player()
        
        # 3. Update world scroll and difficulty
        self._update_world()
        
        # 4. Handle collisions and interactions
        collision_reward, terminated_by_collision = self._handle_collisions()
        reward += collision_reward
        self.score += collision_reward # Add chain reaction points to score
        
        # 5. Update particles
        self._update_particles()
        
        # 6. Check for termination conditions
        terminated = self._check_termination(terminated_by_collision)
        
        # 7. Calculate final reward for the step
        step_reward = self._calculate_reward(terminated)
        reward += step_reward
        self.score += step_reward # Add survival points to score
        
        self.game_over = terminated
        
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL

    def _update_player(self):
        self.player_vel[0] *= self.PLAYER_FRICTION
        self.player_vel[0] = np.clip(self.player_vel[0], -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)
        self.player_pos[0] += self.player_vel[0]
        
        # Clamp player to screen bounds to prevent going off-screen before wall collision
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH)

    def _update_world(self):
        self.world_y += self.game_speed
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.game_speed += 0.05
        
        # Remove off-screen elements
        self.tiles = [tile for tile in self.tiles if tile['world_y'] - self.world_y > -20]

    def _handle_collisions(self):
        reward = 0
        terminated = False
        
        # Wall collision
        left_wall, right_wall = self._get_current_track_bounds()
        player_width = 15
        if self.player_pos[0] - player_width / 2 < left_wall or self.player_pos[0] + player_width / 2 > right_wall:
            # sfx: crash sound
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 50)
            terminated = True
        
        # Tile collision
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 10, 20, 20)
        
        tiles_to_check = list(self.tiles) # Iterate over a copy
        for tile in tiles_to_check:
            tile_y_screen = tile['world_y'] - self.world_y + self.PLAYER_Y_POS
            if abs(tile_y_screen - self.player_pos[1]) < 20:
                tile_rect = pygame.Rect(tile['x'] - tile['size']/2, tile_y_screen - tile['size']/2, tile['size'], tile['size'])
                if player_rect.colliderect(tile_rect):
                    if tile['color_idx'] == self.player_color_idx:
                        chain_reward, chain_len = self._trigger_chain_reaction(tile)
                        reward += 0.5 + chain_reward # Base reward + chain reward
                        # sfx: successful match sound
                        self.game_speed += 0.02 * chain_len # Speed boost
                        # Change player color after a successful match
                        self.player_color_idx = self.np_random.integers(0, len(self.TILE_COLORS))
                    # Remove tile regardless of match to prevent repeated collisions
                    if tile in self.tiles:
                        self.tiles.remove(tile)
                    break # Only one tile interaction per frame
        
        return reward, terminated

    def _trigger_chain_reaction(self, start_tile):
        if start_tile not in self.tiles:
            return 0, 0
            
        q = deque([start_tile])
        visited = { (start_tile['x'], start_tile['world_y']) }
        chain = []
        
        while q:
            current_tile = q.popleft()
            chain.append(current_tile)
            
            # Find adjacent tiles of the same color
            for other_tile in self.tiles:
                if (other_tile['x'], other_tile['world_y']) in visited:
                    continue
                if other_tile['color_idx'] == current_tile['color_idx']:
                    dist_sq = (current_tile['x'] - other_tile['x'])**2 + (current_tile['world_y'] - other_tile['world_y'])**2
                    # Check if they are neighbors (distance is approx tile size)
                    if dist_sq < (current_tile['size'] * 1.5)**2:
                        visited.add((other_tile['x'], other_tile['world_y']))
                        q.append(other_tile)

        chain_len = len(chain)
        for tile in chain:
            if tile in self.tiles:
                self.tiles.remove(tile)
            tile_screen_y = tile['world_y'] - self.world_y + self.PLAYER_Y_POS
            self._create_explosion([tile['x'], tile_screen_y], self.TILE_COLORS[tile['color_idx']], 15)
        
        # Reward: +1 per tile in chain, multiplied by chain length
        chain_reward = chain_len * chain_len
        return chain_reward, chain_len

    def _calculate_reward(self, terminated):
        if terminated:
            if self.world_y >= self.TRACK_LENGTH:
                return 100.0  # Reached finish line
            else:
                return -100.0 # Crashed
        
        # Small reward for moving forward
        return 0.1

    def _check_termination(self, collision_termination):
        return (
            collision_termination or
            self.world_y >= self.TRACK_LENGTH
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
            "speed": self.game_speed
        }
    
    # --- Rendering ---
    
    def _render_game(self):
        self._render_track()
        self._render_tiles()
        self._render_particles()
        self._render_player()
    
    def _render_track(self):
        for i in range(len(self.track) - 1):
            p1 = self.track[i]
            p2 = self.track[i+1]

            p1_y_screen = p1['y'] - self.world_y + self.PLAYER_Y_POS
            p2_y_screen = p2['y'] - self.world_y + self.PLAYER_Y_POS
            
            # Cull off-screen segments
            if p2_y_screen < 0 or p1_y_screen > self.SCREEN_HEIGHT:
                continue

            # Left wall
            pygame.draw.aaline(self.screen, self.COLOR_WALL, (p1['left'], p1_y_screen), (p2['left'], p2_y_screen))
            # Right wall
            pygame.draw.aaline(self.screen, self.COLOR_WALL, (p1['right'], p1_y_screen), (p2['right'], p2_y_screen))

        # Finish line
        finish_y_screen = self.TRACK_LENGTH - self.world_y + self.PLAYER_Y_POS
        if 0 < finish_y_screen < self.SCREEN_HEIGHT:
            left, right = self._get_track_bounds_at_y(self.TRACK_LENGTH)
            for i in range(int(left), int(right), 20):
                color = self.COLOR_FINISH_LINE if (i // 20) % 2 == 0 else self.COLOR_BG
                pygame.draw.rect(self.screen, color, (i, finish_y_screen - 5, 20, 10))

    def _render_tiles(self):
        for tile in self.tiles:
            y_pos = tile['world_y'] - self.world_y + self.PLAYER_Y_POS
            if 0 < y_pos < self.SCREEN_HEIGHT:
                color = self.TILE_COLORS[tile['color_idx']]
                size = tile['size']
                pygame.draw.rect(self.screen, color, (tile['x'] - size/2, y_pos - size/2, size, size), border_radius=4)

    def _render_player(self):
        if self.game_over and self.world_y < self.TRACK_LENGTH:
             return # Don't render crashed player
             
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        player_color = self.TILE_COLORS[self.player_color_idx]

        # Ship body
        points = [
            (x, y - 12),
            (x - 8, y + 8),
            (x + 8, y + 8)
        ]
        
        # Glow effect
        glow_points = [
            (x, y - 18),
            (x - 14, y + 12),
            (x + 14, y + 12)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)

        # Main body
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        
        # Color indicator
        pygame.draw.circle(self.screen, player_color, (x, y + 2), 5)
        pygame.draw.circle(self.screen, (255,255,255), (x, y + 2), 5, 1)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            # This check is needed because pygame surfaces with alpha can't handle alpha=0
            if alpha > 0:
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        speed_text = self.small_font.render(f"SPEED: {self.game_speed:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH - speed_text.get_width() - 10, 10))

    # --- Helper Functions ---

    def _generate_track(self):
        self.track = []
        self.tiles = []
        
        center_x = self.SCREEN_WIDTH / 2
        width = self.SCREEN_WIDTH / 2.5
        
        num_segments = int(self.TRACK_LENGTH / self.TRACK_SEGMENT_LENGTH)
        
        for i in range(num_segments + 2):
            y = i * self.TRACK_SEGMENT_LENGTH
            
            center_x += self.np_random.uniform(-30, 30)
            center_x = np.clip(center_x, width/2 + 50, self.SCREEN_WIDTH - width/2 - 50)
            
            width += self.np_random.uniform(-10, 10)
            width = np.clip(width, 100, self.SCREEN_WIDTH / 2)
            
            self.track.append({
                'y': y,
                'left': center_x - width / 2,
                'right': center_x + width / 2
            })
            
            # Place tiles in this segment
            if i > 5 and i < num_segments - 10: # Don't place near start/end
                if self.np_random.random() < 0.6: # Chance to have a tile pattern
                    self._generate_tile_pattern(y, center_x - width / 2, center_x + width / 2)

    def _generate_tile_pattern(self, world_y, left_bound, right_bound):
        num_lanes = 5
        lane_width = (right_bound - left_bound) / num_lanes
        
        for i in range(num_lanes):
            if self.np_random.random() < 0.4:
                tile_x = left_bound + (i + 0.5) * lane_width
                tile_y_offset = self.np_random.uniform(-self.TRACK_SEGMENT_LENGTH/2, self.TRACK_SEGMENT_LENGTH/2)
                
                self.tiles.append({
                    'x': tile_x,
                    'world_y': world_y + tile_y_offset,
                    'color_idx': self.np_random.integers(0, len(self.TILE_COLORS)),
                    'size': 20
                })

    def _get_current_track_bounds(self):
        return self._get_track_bounds_at_y(self.world_y)

    def _get_track_bounds_at_y(self, y_world):
        segment_idx = int(y_world / self.TRACK_SEGMENT_LENGTH)
        segment_idx = np.clip(segment_idx, 0, len(self.track) - 2)

        p1 = self.track[segment_idx]
        p2 = self.track[segment_idx + 1]
        
        # Interpolate between the two points
        interp_factor = (y_world % self.TRACK_SEGMENT_LENGTH) / self.TRACK_SEGMENT_LENGTH
        
        left = p1['left'] + (p2['left'] - p1['left']) * interp_factor
        right = p1['right'] + (p2['right'] - p1['right']) * interp_factor
        
        return left, right

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] *= 0.98

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'color': color,
                'life': life,
                'max_life': life,
                'size': self.np_random.uniform(2, 6)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == '__main__':
    # This block is for manual play and visualization.
    # It will not be executed by the test suite.
    # To run, you might need to uninstall dummy video driver or set
    # os.environ["SDL_VIDEODRIVER"] = "x11" or "windows" etc.
    # and install pygame.
    try:
        os.environ["SDL_VIDEODRIVER"] = "pygame"
        env = GameEnv(render_mode="rgb_array")
        
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        # Override pygame screen for direct display
        env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Color Racer")
        
        while not terminated and not truncated:
            # Map pygame keys to gymnasium actions
            movement = 0 # no-op
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            if keys[pygame.K_RIGHT]:
                movement = 4
            
            action = [movement, 0, 0] # space/shift not used

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    terminated = True

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # We call _render_game and _render_ui to draw on the display screen
            env.screen.fill(GameEnv.COLOR_BG)
            env._render_game()
            env._render_ui()
            pygame.display.flip()
            
            env.clock.tick(30) # Limit to 30 FPS for playability

        print(f"Game Over. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    except pygame.error as e:
        print(f"Could not run visualization. Pygame error: {e}")
        print("This is normal if you are in a headless environment.")
        print("The code is still valid for training.")
        # Run a validation check in headless mode
        env = GameEnv()
        env.validate_implementation()
    finally:
        if 'env' in locals():
            env.close()