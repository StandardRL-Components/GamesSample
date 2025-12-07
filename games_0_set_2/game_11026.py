import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:27:53.046409
# Source Brief: brief_01026.md
# Brief Index: 1026
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, vel, color, lifespan, size):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.size = size

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color_with_alpha = self.color + (alpha,)
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (self.size, self.size), self.size)
            surface.blit(temp_surf, (int(self.pos[0] - self.size), int(self.pos[1] - self.size)), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Infiltrate a cyberpunk city, changing your color to blend in with the crowd and avoid patrol drones. "
        "Reach the objective before you're detected or run out of time."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press space to change color to match the crowd and become camouflaged. "
        "Press shift to alter special tiles in your path."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    TILE_SIZE = 16
    GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // TILE_SIZE, SCREEN_HEIGHT // TILE_SIZE
    MAX_STEPS = 1000
    PLAYER_SPEED = 0.25 # Interpolation speed
    CROWD_COLOR_CHANGE_INTERVAL = 75 # Steps
    DRONE_DIFFICULTY_INTERVAL = 200 # Steps

    # --- Colors ---
    COLOR_BG = (10, 5, 25)
    COLOR_GRID_LINES = (20, 15, 40)
    COLOR_OBSTACLE = (30, 25, 60)
    COLOR_PATH = (15, 10, 30)
    COLOR_TERRAFORMABLE_OBSTACLE = (50, 45, 90)
    COLOR_TERRAFORMABLE_PATH = (25, 20, 50)
    
    PLAYER_COLORS = [(255, 0, 128), (0, 255, 255), (255, 255, 0)] # Magenta, Cyan, Yellow
    CROWD_COLORS = PLAYER_COLORS
    
    COLOR_AI = (0, 255, 128)
    COLOR_DRONE = (255, 50, 50)
    
    COLOR_UI_TEXT = (200, 200, 220)
    COLOR_UI_BG = (40, 35, 70, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self.grid_map = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.player_pos = np.array([0.0, 0.0])
        self.player_draw_pos = np.array([0.0, 0.0])
        self.player_color_index = 0
        self.player_last_move_dir = np.array([0, 1])
        
        self.ai_pos = np.array([0, 0])
        self.drones = []
        self.crowd = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_dist_to_ai = 0.0
        self.crowd_color_index = 0
        self.crowd_color_timer = 0
        self.base_drone_speed = 0.02
        self.drone_speed_increase = 0.0

        self.particles = []

        # self.reset() is called by the wrapper, no need to call here.
        
    def _create_level(self):
        # 0: Path, 1: Obstacle, 2: Terraformable Path, 3: Terraformable Obstacle
        self.grid_map = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Carve out a path
        path_coords = [
            (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1),
            (8, 2), (8, 3), (8, 4), (8, 5), (7, 5), (6, 5), (5, 5), (4, 5), (3, 5),
            (3, 6), (3, 7), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (10, 8),
            (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1),
            (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1),
            (18, 2), (18, 3), (18, 4), (18, 5), (18, 6), (18, 7), (18, 8), (18, 9), (18, 10),
            (17, 10), (16, 10), (15, 10), (14, 10), (13, 10), (12, 10),
            (12, 11), (12, 12), (12, 13), (12, 14), (13, 14), (14, 14), (15, 14), (16, 14), (17, 14), (18, 14),
            (18, 15), (18, 16), (18, 17), (18, 18), (18, 19),
            (17, 19), (16, 19), (15, 19), (14, 19), (13, 19), (12, 19), (11, 19), (10, 19),
            (10, 20), (10, 21), (10, 22), (10, 23),
            (11, 23), (12, 23), (13, 23), (14, 23), (15, 23), (16, 23), (17, 23), (18, 23),
            (19, 23), (20, 23), (21, 23), (22, 23), (23, 23), (24, 23), (25, 23),
            (25, 22), (25, 21), (25, 20), (25, 19), (25, 18), (25, 17),
            (26, 17), (27, 17), (28, 17), (29, 17), (30, 17), (31, 17), (32, 17),
            (32, 16), (32, 15), (32, 14), (32, 13), (32, 12), (32, 11), (32, 10), (32, 9),
            (33, 9), (34, 9), (35, 9), (36, 9), (37, 9), (38, 9),
            (38, 8), (38, 7), (38, 6), (38, 5), (38, 4), (38, 3), (38, 2), (38, 1)
        ]
        for x, y in path_coords:
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid_map[x, y] = 0

        # Add terraformable zones
        for y in range(11, 19): self.grid_map[22, y] = 2
        for x in range(2, 8): self.grid_map[x, 15] = 2
        for x in range(33, 38): self.grid_map[x, 15] = 2
        self.grid_map[22, 10] = 3
        self.grid_map[22, 19] = 3
        
        self.player_pos = np.array([2.0, 2.0])
        self.player_draw_pos = self.player_pos * self.TILE_SIZE + self.TILE_SIZE / 2
        
        self.ai_pos = np.array([self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2])
        self.grid_map[self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2] = 0

        # Drones with patrol paths
        self.drones = [
            {'pos': np.array([10.0, 15.0]), 'path': [np.array([10, 15]), np.array([18, 15])], 'target_idx': 1, 'speed': 0},
            {'pos': np.array([25.0, 5.0]), 'path': [np.array([25, 5]), np.array([25, 20])], 'target_idx': 1, 'speed': 0},
            {'pos': np.array([5.0, 20.0]), 'path': [np.array([5, 20]), np.array([35, 20])], 'target_idx': 1, 'speed': 0}
        ]
        
        # Crowd with movement patterns
        self.crowd = []
        for i in range(20):
            start_x = self.np_random.integers(1, self.GRID_WIDTH - 1)
            start_y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
            if self.grid_map[start_x, start_y] == 0:
                 self.crowd.append({'pos': np.array([float(start_x), float(start_y)]), 'offset': self.np_random.random() * 2 * math.pi})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._create_level()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_color_index = self.np_random.integers(0, len(self.PLAYER_COLORS))
        self.player_last_move_dir = np.array([0, 1])
        
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        self.last_dist_to_ai = self._get_dist_to_ai()
        
        self.crowd_color_index = self.np_random.integers(0, len(self.CROWD_COLORS))
        self.crowd_color_timer = 0
        
        self.base_drone_speed = 0.02
        self.drone_speed_increase = 0.0
        for drone in self.drones:
            drone['speed'] = self.base_drone_speed

        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions ---
        reward += self._handle_player_movement(movement)
        reward += self._handle_color_change(space_held)
        reward += self._handle_terraform(shift_held)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self.steps += 1
        self.crowd_color_timer += 1
        if self.crowd_color_timer >= self.CROWD_COLOR_CHANGE_INTERVAL:
            self.crowd_color_timer = 0
            self.crowd_color_index = (self.crowd_color_index + 1) % len(self.CROWD_COLORS)
            # sfx: crowd_color_change_chime.wav
        
        self._update_difficulty()
        self._update_drones()
        self._update_crowd()
        self._update_particles()
        
        # --- Calculate Rewards & Check Termination ---
        # Proximity reward
        dist_to_ai = self._get_dist_to_ai()
        reward += (self.last_dist_to_ai - dist_to_ai) * 0.1
        self.last_dist_to_ai = dist_to_ai

        # Detection check
        detection_reward, is_detected = self._check_drone_detection()
        reward += detection_reward
        if is_detected:
            terminated = True
            # sfx: alarm.wav, player_detected.wav
        
        # Goal check
        if np.array_equal(self.player_pos.astype(int), self.ai_pos):
            reward += 100
            terminated = True
            # sfx: level_complete.wav
        
        # Step limit
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_movement(self, movement_action):
        move_dir = np.array([0, 0])
        if movement_action == 1: move_dir = np.array([0, -1]) # Up
        elif movement_action == 2: move_dir = np.array([0, 1]) # Down
        elif movement_action == 3: move_dir = np.array([-1, 0]) # Left
        elif movement_action == 4: move_dir = np.array([1, 0]) # Right

        if np.any(move_dir):
            self.player_last_move_dir = move_dir
            
            target_pos = self.player_pos + move_dir
            target_x, target_y = int(target_pos[0]), int(target_pos[1])

            if 0 <= target_x < self.GRID_WIDTH and 0 <= target_y < self.GRID_HEIGHT:
                tile_type = self.grid_map[target_x, target_y]
                if tile_type == 0 or tile_type == 2: # Path or Terraformable Path
                    self.player_pos = target_pos
                    # sfx: player_step.wav
        return 0

    def _handle_color_change(self, space_held):
        reward = 0
        if space_held and not self.prev_space_held:
            self.player_color_index = (self.player_color_index + 1) % len(self.PLAYER_COLORS)
            # sfx: color_change.wav
            
            # Create particle effect
            player_center = self.player_draw_pos
            new_color = self.PLAYER_COLORS[self.player_color_index]
            for _ in range(30):
                angle = self.np_random.random() * 2 * math.pi
                speed = self.np_random.random() * 2 + 1
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifespan = self.np_random.integers(15, 30)
                size = self.np_random.integers(2, 5)
                self.particles.append(Particle(player_center, vel, new_color, lifespan, size))

            # Reward for matching crowd color
            if self.player_color_index == self.crowd_color_index:
                reward += 5.0
        return reward

    def _handle_terraform(self, shift_held):
        if shift_held and not self.prev_shift_held:
            target_pos = self.player_pos + self.player_last_move_dir
            target_x, target_y = int(target_pos[0]), int(target_pos[1])

            if 0 <= target_x < self.GRID_WIDTH and 0 <= target_y < self.GRID_HEIGHT:
                tile_type = self.grid_map[target_x, target_y]
                if tile_type == 2: # Terraformable Path -> Obstacle
                    self.grid_map[target_x, target_y] = 3
                    self._create_terraform_effect(target_x, target_y, self.COLOR_TERRAFORMABLE_OBSTACLE)
                    # sfx: terraform_on.wav
                elif tile_type == 3: # Terraformable Obstacle -> Path
                    self.grid_map[target_x, target_y] = 2
                    self._create_terraform_effect(target_x, target_y, self.COLOR_TERRAFORMABLE_PATH)
                    # sfx: terraform_off.wav
        return 0

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.DRONE_DIFFICULTY_INTERVAL == 0:
            self.drone_speed_increase += 0.01
            for drone in self.drones:
                drone['speed'] = self.base_drone_speed + self.drone_speed_increase

    def _update_drones(self):
        for drone in self.drones:
            target_pos = drone['path'][drone['target_idx']]
            direction = target_pos - drone['pos']
            dist = np.linalg.norm(direction)
            
            if dist < 0.1:
                drone['target_idx'] = 1 - drone['target_idx']
            else:
                drone['pos'] += (direction / dist) * drone['speed']

    def _update_crowd(self):
        for member in self.crowd:
            # Simple circular motion for ambient movement
            time_factor = (self.steps / 100.0) + member['offset']
            member['pos'][0] += math.sin(time_factor) * 0.02
            member['pos'][1] += math.cos(time_factor * 0.8) * 0.02

            # Clamp to screen
            member['pos'][0] = np.clip(member['pos'][0], 0, self.GRID_WIDTH - 1)
            member['pos'][1] = np.clip(member['pos'][1], 0, self.GRID_HEIGHT - 1)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _check_drone_detection(self):
        is_camouflaged = self.player_color_index == self.crowd_color_index
        reward = 0
        is_detected = False
        for drone in self.drones:
            dist = np.linalg.norm(self.player_pos - drone['pos'])
            if dist < 2.0 and not is_camouflaged:
                reward -= 0.5 # Continuous penalty for being close and un-camouflaged
            if dist < 1.5 and not is_camouflaged:
                reward -= 10.0 # Caught!
                is_detected = True
                break
        return reward, is_detected

    def _get_dist_to_ai(self):
        return np.linalg.norm(self.player_pos - self.ai_pos)

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
            "player_pos": self.player_pos,
            "dist_to_ai": self.last_dist_to_ai,
        }

    def _render_game(self):
        # Interpolate player draw position for smooth movement
        target_draw_pos = self.player_pos * self.TILE_SIZE + self.TILE_SIZE / 2
        self.player_draw_pos = self.player_draw_pos * (1 - self.PLAYER_SPEED) + target_draw_pos * self.PLAYER_SPEED

        # Draw grid
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                tile_type = self.grid_map[x, y]
                color = self.COLOR_PATH
                if tile_type == 1: color = self.COLOR_OBSTACLE
                elif tile_type == 2: color = self.COLOR_TERRAFORMABLE_PATH
                elif tile_type == 3: color = self.COLOR_TERRAFORMABLE_OBSTACLE
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw background details
        self._draw_background_details()

        # Draw crowd
        crowd_color = self.CROWD_COLORS[self.crowd_color_index]
        for member in self.crowd:
            pos_x = int(member['pos'][0] * self.TILE_SIZE + self.TILE_SIZE / 2)
            pos_y = int(member['pos'][1] * self.TILE_SIZE + self.TILE_SIZE / 2)
            pygame.draw.circle(self.screen, crowd_color, (pos_x, pos_y), 3)

        # Draw AI
        ai_px_pos = self.ai_pos * self.TILE_SIZE + self.TILE_SIZE / 2
        self._draw_glow(self.screen, ai_px_pos.astype(int), self.COLOR_AI, 20)
        pygame.draw.rect(self.screen, self.COLOR_AI, (ai_px_pos[0]-4, ai_px_pos[1]-4, 8, 8))

        # Draw drones
        for drone in self.drones:
            drone_px_pos = drone['pos'] * self.TILE_SIZE + self.TILE_SIZE / 2
            self._draw_glow(self.screen, drone_px_pos.astype(int), self.COLOR_DRONE, 25)
            p = [
                (drone_px_pos[0], drone_px_pos[1] - 6),
                (drone_px_pos[0] - 5, drone_px_pos[1] + 4),
                (drone_px_pos[0] + 5, drone_px_pos[1] + 4)
            ]
            pygame.gfxdraw.aapolygon(self.screen, p, self.COLOR_DRONE)
            pygame.gfxdraw.filled_polygon(self.screen, p, self.COLOR_DRONE)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw player
        player_color = self.PLAYER_COLORS[self.player_color_index]
        self._draw_glow(self.screen, self.player_draw_pos.astype(int), player_color, 20)
        player_rect = (self.player_draw_pos[0]-4, self.player_draw_pos[1]-4, 8, 8)
        pygame.draw.rect(self.screen, player_color, player_rect)
        pygame.draw.rect(self.screen, (255,255,255), player_rect, 1)

    def _render_ui(self):
        # UI Background
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, self.SCREEN_HEIGHT - 40))

        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, self.SCREEN_HEIGHT - 32))

        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 30))

        # Color status
        status_text = self.font_small.render("STATUS:", True, self.COLOR_UI_TEXT)
        self.screen.blit(status_text, (250, self.SCREEN_HEIGHT - 30))
        
        # Player color indicator
        player_color_rect = (320, self.SCREEN_HEIGHT - 32, 24, 24)
        pygame.draw.rect(self.screen, self.PLAYER_COLORS[self.player_color_index], player_color_rect)
        pygame.draw.rect(self.screen, (255,255,255), player_color_rect, 2)
        
        # Crowd color indicator
        crowd_color_rect = (355, self.SCREEN_HEIGHT - 32, 24, 24)
        pygame.draw.rect(self.screen, self.CROWD_COLORS[self.crowd_color_index], crowd_color_rect)
        pygame.draw.rect(self.screen, (100,100,100), crowd_color_rect, 2)

        # Camouflage status
        is_camouflaged = self.player_color_index == self.crowd_color_index
        cam_text = "CAMOUFLAGED" if is_camouflaged else "EXPOSED"
        cam_color = (0, 255, 128) if is_camouflaged else (255, 50, 50)
        cam_surf = self.font_small.render(cam_text, True, cam_color)
        self.screen.blit(cam_surf, (390, self.SCREEN_HEIGHT - 30))

    def _draw_glow(self, surface, pos, color, radius):
        for i in range(radius // 2, 0, -2):
            alpha = 60 * (1 - (i / (radius // 2)))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], i, color + (int(alpha),))
    
    def _draw_background_details(self):
        # Flashing lights on buildings when player is detected
        is_detected = any(np.linalg.norm(self.player_pos - d['pos']) < 1.5 and self.player_color_index != self.crowd_color_index for d in self.drones)
        
        if is_detected and self.steps % 10 < 5:
            flash_color = (255, 0, 0, 100)
            for x in range(1, self.GRID_WIDTH, 5):
                for y in range(1, self.GRID_HEIGHT, 5):
                    if self.grid_map[x,y] == 1:
                        pygame.draw.rect(self.screen, flash_color, (x*self.TILE_SIZE, y*self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE), border_radius=2)

    def _create_terraform_effect(self, grid_x, grid_y, color):
        center_pos = (grid_x * self.TILE_SIZE + self.TILE_SIZE / 2, grid_y * self.TILE_SIZE + self.TILE_SIZE / 2)
        for _ in range(40):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 1.5 + 0.5
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            size = self.np_random.integers(1, 4)
            self.particles.append(Particle(center_pos, vel, color, lifespan, size))

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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv()
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Cyberpunk Stealth")
        clock = pygame.time.Clock()
        
        terminated = False
        
        # --- Action mapping for human play ---
        # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        # actions[1]: Space button (0=released, 1=held)
        # actions[2]: Shift button (0=released, 1=held)
        action = [0, 0, 0]

        while not terminated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            
            # Reset action
            action = [0, 0, 0]

            # Movement
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                action[0] = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                action[0] = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                action[0] = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                action[0] = 4
            
            # Other actions
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Run at 30 FPS

        print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
        env.close()
    except pygame.error as e:
        print(f"Could not initialize display for manual play: {e}")
        print("This is expected in a headless environment. The environment itself is likely working correctly.")
        # Create an instance to run validation in headless mode
        print("Running validation...")
        env = GameEnv()
        env.reset()
        env.validate_implementation()
        env.close()