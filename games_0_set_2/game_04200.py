# Generated: 2025-08-28T01:42:51.846464
# Source Brief: brief_04200.md
# Brief Index: 4200

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set headless mode for Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"


class Particle:
    def __init__(self, x, y, vx, vy, color, radius, gravity, lifetime):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.radius = radius
        self.gravity = gravity
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        self.vy += self.gravity
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1

    def draw(self, surface, camera_offset):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            # The color is expected to be a tuple (R, G, B) to which we concatenate another tuple (A,).
            color = self.color + (alpha,)
            current_radius = int(self.radius * (self.lifetime / self.initial_lifetime))
            if current_radius > 0:
                pos = (int(self.x - camera_offset[0]), int(self.y - camera_offset[1]))
                pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], current_radius, color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move. ↑ or Space to jump. Reach the green flag!"
    )

    game_description = (
        "A fast-paced 2D platformer. Guide the robot across procedurally generated "
        "levels, avoiding obstacles and racing against the clock to reach the goal."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Pygame Setup
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)
        self.COLOR_BG_BOTTOM = (0, 25, 50)
        self.COLOR_PLAYER = (65, 105, 225) # Royal Blue
        self.COLOR_PLAYER_EYE = (255, 255, 255)
        self.COLOR_PLATFORM = (100, 100, 110)
        self.COLOR_PLATFORM_OUTLINE = (60, 60, 70)
        self.COLOR_OBSTACLE = (255, 69, 0) # Red-Orange
        self.COLOR_GOAL = (50, 205, 50) # Lime Green
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_JUMP_PARTICLE = (255, 215, 0) # Gold
        self.COLOR_LAND_PARTICLE = (200, 200, 200)

        # Physics & Player Constants
        self.GRAVITY = 0.4
        self.JUMP_STRENGTH = -10
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = 0.88
        self.MAX_SPEED_X = 6
        self.PLAYER_WIDTH = 24
        self.PLAYER_HEIGHT = 36
        
        # Game Constants
        self.STAGE_TIME = 60 * 30 # 60 seconds at 30 FPS
        self.MAX_STAGES = 3
        self.NUM_PLATFORMS_PER_STAGE = 15

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.platforms = None
        self.obstacles = None
        self.end_flag_rect = None
        self.particles = None
        self.camera_offset = None
        self.stage = None
        self.time_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.last_player_x = None
        self.last_platform_landed_idx = None
        
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.camera_offset = [0, 0]
        
        self._setup_stage(1)
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self, stage_num):
        self.stage = stage_num
        self.time_left = self.STAGE_TIME
        self.platforms = []
        self.obstacles = []
        
        start_platform = pygame.Rect(50, self.HEIGHT - 80, 200, 80)
        self.platforms.append(start_platform)
        
        self.player_pos = [start_platform.centerx, start_platform.top - self.PLAYER_HEIGHT]
        self.player_vel = [0, 0]
        self.on_ground = True
        self.last_player_x = self.player_pos[0]
        self.last_platform_landed_idx = 0

        current_x = start_platform.right
        current_y = start_platform.y
        
        for i in range(self.NUM_PLATFORMS_PER_STAGE):
            gap = self.np_random.integers(60, 120) * (1 + 0.05 * (self.stage - 1))
            width = self.np_random.integers(100, 300)
            height = self.np_random.integers(40, 120)
            
            y_change = self.np_random.integers(-80, 80)
            next_y = np.clip(current_y + y_change, 150, self.HEIGHT - height)
            
            platform = pygame.Rect(current_x + gap, next_y, width, height)
            self.platforms.append(platform)
            
            # Add obstacles randomly
            if self.np_random.random() < 0.25 and i > 0:
                obstacle_size = 20
                ox = platform.x + self.np_random.integers(obstacle_size, platform.width - obstacle_size)
                oy = platform.top - obstacle_size
                self.obstacles.append(pygame.Rect(ox, oy, obstacle_size, obstacle_size))
            
            current_x = platform.right
            current_y = platform.y

        last_platform = self.platforms[-1]
        flag_pole_height = 60
        self.end_flag_rect = pygame.Rect(last_platform.centerx - 15, last_platform.top - flag_pole_height, 30, 20)
        self.end_flag_pole_rect = pygame.Rect(last_platform.centerx, last_platform.top - flag_pole_height, 4, flag_pole_height)


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0
        self.steps += 1
        self.time_left -= 1

        # --- Handle Input ---
        is_jumping = (movement == 1 or space_held)
        if is_jumping and self.on_ground:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: Jump
            for _ in range(10):
                self.particles.append(Particle(
                    self.player_pos[0] + self.PLAYER_WIDTH / 2, self.player_pos[1] + self.PLAYER_HEIGHT,
                    (self.np_random.random() - 0.5) * 3, self.np_random.random() * -2,
                    self.COLOR_JUMP_PARTICLE, 5, 0.1, 20
                ))

        accel_x = 0
        if movement == 3: # Left
            accel_x = -self.PLAYER_ACCEL
        elif movement == 4: # Right
            accel_x = self.PLAYER_ACCEL
        
        # --- Update Physics ---
        self.player_vel[0] += accel_x
        if accel_x == 0:
            self.player_vel[0] *= self.PLAYER_FRICTION
        self.player_vel[0] = np.clip(self.player_vel[0], -self.MAX_SPEED_X, self.MAX_SPEED_X)
        self.player_vel[1] += self.GRAVITY
        
        # --- Collision Detection and Resolution ---
        self.player_pos[0] += self.player_vel[0]
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        for platform in self.platforms:
            if player_rect.colliderect(platform):
                if self.player_vel[0] > 0:
                    player_rect.right = platform.left
                    self.player_vel[0] = 0
                elif self.player_vel[0] < 0:
                    player_rect.left = platform.right
                    self.player_vel[0] = 0
                self.player_pos[0] = player_rect.x

        self.player_pos[1] += self.player_vel[1]
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        self.on_ground = False
        
        for i, platform in enumerate(self.platforms):
            if player_rect.colliderect(platform):
                if self.player_vel[1] > 0 and player_rect.bottom < platform.bottom:
                    player_rect.bottom = platform.top
                    self.player_vel[1] = 0
                    self.on_ground = True
                    if self.last_platform_landed_idx != i:
                        self.last_platform_landed_idx = i
                        reward += 1.0
                        self.score += 10
                        # sfx: Land
                        for _ in range(5):
                            self.particles.append(Particle(
                                player_rect.midbottom[0], player_rect.midbottom[1],
                                (self.np_random.random() - 0.5) * 4, self.np_random.random() * -1,
                                self.COLOR_LAND_PARTICLE, 3, 0.2, 15
                            ))
                elif self.player_vel[1] < 0:
                    player_rect.top = platform.bottom
                    self.player_vel[1] = 0
                self.player_pos[1] = player_rect.y

        # --- Update Particles ---
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]

        # --- Reward for progress ---
        if self.player_pos[0] > self.last_player_x:
            reward += 0.1 * (self.player_pos[0] - self.last_player_x) / self.MAX_SPEED_X
        else:
            reward -= 0.01
        self.last_player_x = self.player_pos[0]
        
        # --- Check Termination Conditions ---
        terminated = False
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

        # Fall off screen
        if player_rect.top > self.HEIGHT:
            reward -= 10
            terminated = True
            # sfx: Fall/Fail

        # Time out
        if self.time_left <= 0:
            reward -= 10
            terminated = True
            # sfx: Timeout

        # Hit obstacle
        for obs in self.obstacles:
            if player_rect.colliderect(obs):
                reward -= 10
                terminated = True
                # sfx: Hit/Hurt

        # Reached goal
        if player_rect.colliderect(self.end_flag_rect):
            if self.stage < self.MAX_STAGES:
                self.score += 100
                # sfx: Stage Clear
                self._setup_stage(self.stage + 1)
            else:
                reward += 100
                self.score += 250
                terminated = True
                # sfx: Win
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left": self.time_left // 30,
        }

    def _get_observation(self):
        # Update camera
        target_cam_x = self.player_pos[0] - self.WIDTH / 2.5
        target_cam_y = self.player_pos[1] - self.HEIGHT / 1.5
        self.camera_offset[0] += (target_cam_x - self.camera_offset[0]) * 0.1
        self.camera_offset[1] += (target_cam_y - self.camera_offset[1]) * 0.1

        self._render_background()
        self._render_platforms()
        self._render_obstacles()
        self._render_goal()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            mix_ratio = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - mix_ratio) + self.COLOR_BG_BOTTOM[i] * mix_ratio)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_platforms(self):
        for p in self.platforms:
            cam_p = p.move(-self.camera_offset[0], -self.camera_offset[1])
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, cam_p)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, cam_p.inflate(-4, -4))

    def _render_obstacles(self):
        for obs in self.obstacles:
            cam_obs = obs.move(-self.camera_offset[0], -self.camera_offset[1])
            p1 = cam_obs.midtop
            p2 = cam_obs.bottomleft
            p3 = cam_obs.bottomright
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_OBSTACLE)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_OBSTACLE)

    def _render_goal(self):
        cam_pole = self.end_flag_pole_rect.move(-self.camera_offset[0], -self.camera_offset[1])
        cam_flag = self.end_flag_rect.move(-self.camera_offset[0], -self.camera_offset[1])
        pygame.draw.rect(self.screen, (200, 200, 200), cam_pole)
        pygame.gfxdraw.filled_polygon(self.screen, 
            [cam_flag.topleft, cam_flag.topright, cam_flag.midleft], self.COLOR_GOAL)
        pygame.gfxdraw.aapolygon(self.screen, 
            [cam_flag.topleft, cam_flag.topright, cam_flag.midleft], self.COLOR_GOAL)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen, self.camera_offset)

    def _render_player(self):
        player_rect = pygame.Rect(
            self.player_pos[0] - self.camera_offset[0],
            self.player_pos[1] - self.camera_offset[1],
            self.PLAYER_WIDTH, self.PLAYER_HEIGHT
        )
        
        # Body
        body_rect = player_rect.inflate(-4, 0)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=4)
        
        # Head/Eye
        eye_dir = np.sign(self.player_vel[0]) if self.player_vel[0] != 0 else 1
        eye_pos = (
            int(player_rect.centerx + eye_dir * (self.PLAYER_WIDTH / 4)),
            int(player_rect.centery - self.PLAYER_HEIGHT / 6)
        )
        pygame.gfxdraw.aacircle(self.screen, eye_pos[0], eye_pos[1], 4, self.COLOR_PLAYER_EYE)
        pygame.gfxdraw.filled_circle(self.screen, eye_pos[0], eye_pos[1], 4, self.COLOR_PLAYER_EYE)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        time_text = self.font_small.render(f"TIME: {max(0, self.time_left // 30)}", True, self.COLOR_TEXT)
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        self.screen.blit(stage_text, (10, 30))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        
        message = "YOU WIN!" if self.stage == self.MAX_STAGES and self.time_left > 0 and self.player_pos[1] < self.HEIGHT else "GAME OVER"
        text = self.font_large.render(message, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You might need to change 'x11' to 'windows' or 'cocoa' depending on your OS
    # or remove it if you are running in a headless environment.
    os.environ['SDL_VIDEODRIVER'] = 'x11' 

    env = GameEnv()
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Robo-Jumper")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    print(env.user_guide)
    
    while running:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_UP]:
            movement = 1
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS

    env.close()