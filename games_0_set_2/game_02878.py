
# Generated: 2025-08-27T21:42:08.702175
# Source Brief: brief_02878.md
# Brief Index: 2878

        
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
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    user_guide = (
        "Controls: Use arrow keys to select the direction of your next hop. "
        "The ship will automatically jump to the nearest platform in that direction."
    )

    game_description = (
        "Navigate a spaceship through a treacherous asteroid field. "
        "Hop between platforms to reach the goal before time runs out. "
        "Faster completion and risky jumps earn more points."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TIME_LIMIT_SECONDS = 60
    FPS = 30
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_PLATFORM = (60, 120, 255)
    COLOR_PLATFORM_START = (100, 200, 255)
    COLOR_PLATFORM_END = (255, 220, 0)
    COLOR_ASTEROID = (255, 50, 50)
    COLOR_ASTEROID_OUTLINE = (255, 150, 150)
    COLOR_STAR = (200, 200, 220)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # Game Object Properties
    PLAYER_SIZE = 10
    HOP_DURATION_FRAMES = 12
    PLATFORM_WIDTH = 45
    PLATFORM_HEIGHT = 12
    NUM_PLATFORMS = 12
    NUM_ASTEROIDS = 15
    ASTEROID_BASE_RADIUS = 15
    ASTEROID_JAGGEDNESS = 6
    INITIAL_ASTEROID_SPEED = 1.0
    ASTEROID_SPEED_INCREASE_INTERVAL = 1000
    ASTEROID_SPEED_INCREASE_AMOUNT = 0.05
    NUM_STARS = 150

    # Reward Radii
    NEAR_MISS_RADIUS = 50
    PROXIMITY_PENALTY_RADIUS = 100

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
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 36)

        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_hop_start_pos = np.zeros(2, dtype=np.float32)
        self.player_hop_target_pos = np.zeros(2, dtype=np.float32)
        self.player_hop_progress = 1.0
        self.platforms = []
        self.current_platform_index = 0
        self.asteroids = []
        self.stars = []
        self.base_asteroid_speed = self.INITIAL_ASTEROID_SPEED
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.win_message = ""
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.base_asteroid_speed = self.INITIAL_ASTEROID_SPEED
        self.win_message = ""

        self._generate_stars()
        self._generate_platforms()
        
        start_platform_pos = self.platforms[0]['pos']
        self.player_pos = np.array(start_platform_pos, dtype=np.float32)
        self.player_hop_start_pos = np.array(start_platform_pos, dtype=np.float32)
        self.player_hop_target_pos = np.array(start_platform_pos, dtype=np.float32)
        self.player_hop_progress = 1.0
        self.current_platform_index = 0

        self._generate_asteroids()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.timer -= 1

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % self.ASTEROID_SPEED_INCREASE_INTERVAL == 0:
            self.base_asteroid_speed += self.ASTEROID_SPEED_INCREASE_AMOUNT
            # sound: difficulty_increase.wav

        # --- Handle Action ---
        movement = action[0]
        hop_initiated = False
        if movement > 0 and self.player_hop_progress >= 1.0:
            target_platform_idx = self._find_target_platform(movement)
            if target_platform_idx is not None and target_platform_idx != self.current_platform_index:
                hop_initiated = True
                self.player_hop_start_pos = np.array(self.platforms[self.current_platform_index]['pos'], dtype=np.float32)
                self.player_hop_target_pos = np.array(self.platforms[target_platform_idx]['pos'], dtype=np.float32)
                self.player_hop_progress = 0.0
                self.current_platform_index = target_platform_idx
                # sound: player_hop.wav

                # --- Hopping Rewards ---
                reward += 10.0  # Successful hop
                
                # Reward for moving towards goal
                end_platform_pos = self.platforms[-1]['pos']
                dist_old = np.linalg.norm(self.player_hop_start_pos - end_platform_pos)
                dist_new = np.linalg.norm(self.player_hop_target_pos - end_platform_pos)
                if dist_new < dist_old:
                    reward += 0.5
                
                # Penalty for choosing platform near asteroids
                for asteroid in self.asteroids:
                    dist_to_asteroid = np.linalg.norm(self.player_hop_target_pos - asteroid['pos'])
                    if dist_to_asteroid < self.PROXIMITY_PENALTY_RADIUS:
                        reward -= 0.2

        # --- Update Game State ---
        self._update_player_hop()
        self._update_asteroids()
        self._update_stars()

        # --- Survival and Proximity Rewards ---
        reward += 0.1  # Survived one step
        
        # Near miss penalty
        for asteroid in self.asteroids:
            dist_to_player = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist_to_player < self.NEAR_MISS_RADIUS and not hop_initiated:
                reward -= 5.0 # Near miss with asteroid
                # sound: near_miss.wav

        self.score += reward
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        # 1. Asteroid Collision
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_SIZE / 2 + asteroid['radius']:
                self.game_over = True
                self.score -= 100
                self.win_message = "GAME OVER"
                # sound: explosion.wav
                return True

        # 2. Time Out
        if self.timer <= 0:
            self.game_over = True
            self.score -= 50
            self.win_message = "TIME'S UP"
            # sound: time_out.wav
            return True

        # 3. Reached End Platform
        if self.current_platform_index == len(self.platforms) - 1 and self.player_hop_progress >= 1.0:
            self.game_over = True
            self.score += 100
            self.win_message = "YOU WIN!"
            # sound: victory.wav
            return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer}
    
    # --- Update Logic ---
    def _update_player_hop(self):
        if self.player_hop_progress < 1.0:
            self.player_hop_progress += 1.0 / self.HOP_DURATION_FRAMES
            self.player_hop_progress = min(1.0, self.player_hop_progress)
            
            # Ease-in-out interpolation
            t = self.player_hop_progress
            smooth_t = t * t * (3.0 - 2.0 * t)
            self.player_pos = self.player_hop_start_pos + (self.player_hop_target_pos - self.player_hop_start_pos) * smooth_t
        else:
            self.player_pos = self.player_hop_target_pos

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel'] * self.base_asteroid_speed
            # Screen wrap
            if asteroid['pos'][0] < -asteroid['radius']: asteroid['pos'][0] = self.SCREEN_WIDTH + asteroid['radius']
            if asteroid['pos'][0] > self.SCREEN_WIDTH + asteroid['radius']: asteroid['pos'][0] = -asteroid['radius']
            if asteroid['pos'][1] < -asteroid['radius']: asteroid['pos'][1] = self.SCREEN_HEIGHT + asteroid['radius']
            if asteroid['pos'][1] > self.SCREEN_HEIGHT + asteroid['radius']: asteroid['pos'][1] = -asteroid['radius']

    def _update_stars(self):
        for star in self.stars:
            star['pos'][0] -= star['speed']
            if star['pos'][0] < 0:
                star['pos'][0] = self.SCREEN_WIDTH
                star['pos'][1] = self.np_random.uniform(0, self.SCREEN_HEIGHT)

    # --- Generation Logic ---
    def _generate_stars(self):
        self.stars = []
        for _ in range(self.NUM_STARS):
            speed = self.np_random.uniform(0.1, 0.7)
            self.stars.append({
                'pos': [self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)],
                'size': int(speed * 3),
                'speed': speed
            })
    
    def _generate_platforms(self):
        self.platforms = []
        start_x, end_x = 80, self.SCREEN_WIDTH - 80
        
        # Start platform
        self.platforms.append({'pos': (start_x, self.SCREEN_HEIGHT / 2), 'type': 'start'})
        
        # Intermediate platforms
        for i in range(1, self.NUM_PLATFORMS - 1):
            px = start_x + i * (end_x - start_x) / (self.NUM_PLATFORMS - 1)
            px += self.np_random.uniform(-20, 20)
            py = self.SCREEN_HEIGHT / 2 + self.np_random.uniform(-120, 120)
            self.platforms.append({'pos': (px, py), 'type': 'normal'})
        
        # End platform
        self.platforms.append({'pos': (end_x, self.SCREEN_HEIGHT / 2), 'type': 'end'})

    def _generate_asteroids(self):
        self.asteroids = []
        start_platform_pos = self.platforms[0]['pos']
        for _ in range(self.NUM_ASTEROIDS):
            while True:
                pos = self.np_random.uniform([0, 0], [self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
                if np.linalg.norm(pos - start_platform_pos) > 100: # Don't spawn on player
                    break
            
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)])
            
            radius = self.ASTEROID_BASE_RADIUS + self.np_random.uniform(-3, 3)
            shape_points = []
            num_verts = 12
            for i in range(num_verts):
                a = (i / num_verts) * 2 * math.pi
                r = radius + self.np_random.uniform(-self.ASTEROID_JAGGEDNESS, self.ASTEROID_JAGGEDNESS)
                shape_points.append((r * math.cos(a), r * math.sin(a)))

            self.asteroids.append({'pos': pos, 'vel': vel, 'radius': radius, 'shape': shape_points})

    # --- Helper Logic ---
    def _find_target_platform(self, direction):
        # 1: up, 2: down, 3: left, 4: right
        direction_vectors = {
            1: np.array([0, -1]), 2: np.array([0, 1]),
            3: np.array([-1, 0]), 4: np.array([1, 0])
        }
        target_dir_vec = direction_vectors[direction]
        
        best_score = -1
        best_idx = None
        
        current_pos = self.platforms[self.current_platform_index]['pos']

        for i, platform in enumerate(self.platforms):
            if i == self.current_platform_index:
                continue

            platform_pos = platform['pos']
            vec_to_platform = np.array(platform_pos) - np.array(current_pos)
            dist = np.linalg.norm(vec_to_platform)
            if dist == 0: continue

            norm_vec_to_platform = vec_to_platform / dist
            
            # Dot product gives cosine of angle between vectors (alignment)
            dot_product = np.dot(target_dir_vec, norm_vec_to_platform)

            # We want platforms in the right general direction (dot > 0)
            # and we prefer closer platforms with good alignment.
            if dot_product > 0.3: # Must be somewhat aligned
                score = dot_product / (dist ** 0.5) # Prioritize alignment, then distance
                if score > best_score:
                    best_score = score
                    best_idx = i
        
        return best_idx

    # --- Rendering Logic ---
    def _render_game(self):
        # Stars
        for star in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (int(star['pos'][0]), int(star['pos'][1])), star['size'])
            
        # Platforms
        for i, platform in enumerate(self.platforms):
            px, py = platform['pos']
            rect = pygame.Rect(px - self.PLATFORM_WIDTH / 2, py - self.PLATFORM_HEIGHT / 2, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
            
            color = self.COLOR_PLATFORM
            if platform['type'] == 'start': color = self.COLOR_PLATFORM_START
            elif platform['type'] == 'end': color = self.COLOR_PLATFORM_END
            
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            
            # Pulsating highlight for current platform
            if i == self.current_platform_index and self.player_hop_progress >= 1.0:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                highlight_color = (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50))
                pygame.draw.rect(self.screen, highlight_color, rect.inflate(pulse * 6, pulse * 6), width=2, border_radius=5)

        # Asteroids
        for asteroid in self.asteroids:
            points = [(p[0] + asteroid['pos'][0], p[1] + asteroid['pos'][1]) for p in asteroid['shape']]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID_OUTLINE)

        # Player
        if not self.game_over:
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            
            # Glow effect
            for i in range(4, 0, -1):
                pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_SIZE + i * 3, (self.COLOR_PLAYER_GLOW[0], self.COLOR_PLAYER_GLOW[1], self.COLOR_PLAYER_GLOW[2], self.COLOR_PLAYER_GLOW[3] // i))
            
            # Rotation
            angle = math.pi / 2 # Default up
            if self.player_hop_progress < 1.0:
                dx = self.player_hop_target_pos[0] - self.player_hop_start_pos[0]
                dy = self.player_hop_target_pos[1] - self.player_hop_start_pos[1]
                angle = math.atan2(-dy, dx) - math.pi / 2
            
            s, c = math.sin(angle), math.cos(angle)
            p1 = (px + self.PLAYER_SIZE * s, py + self.PLAYER_SIZE * c)
            p2 = (px - self.PLAYER_SIZE * 0.5 * s - self.PLAYER_SIZE * 0.5 * c, py - self.PLAYER_SIZE * 0.5 * c + self.PLAYER_SIZE * 0.5 * s)
            p3 = (px - self.PLAYER_SIZE * 0.5 * s + self.PLAYER_SIZE * 0.5 * c, py - self.PLAYER_SIZE * 0.5 * c - self.PLAYER_SIZE * 0.5 * s)
            
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_text(self, text, font, position, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect(center=position)
        shadow_rect = shadow_surf.get_rect(center=(position[0]+2, position[1]+2))
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {int(self.score)}", self.font_small, (100, 30))
        # Timer
        time_left = max(0, self.timer / self.FPS)
        self._render_text(f"TIME: {time_left:.1f}", self.font_small, (self.SCREEN_WIDTH - 100, 30))
        # Game Over Message
        if self.game_over:
            self._render_text(self.win_message, self.font_large, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs, _ = self.reset()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully")


if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    pygame.display.set_caption("Platform Hopper")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and Shift not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f"Episode finished. Score: {info['score']:.2f}. Press 'R' to reset.")

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(env.FPS)

    env.close()