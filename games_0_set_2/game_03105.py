import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to jump, ←→ to move. Collect coins and reach the end of each stage."
    )

    game_description = (
        "Guide a hopping spaceship through obstacle-ridden space, collecting coins and reaching the end of each stage."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (13, 7, 21)
    COLOR_PLAYER = (66, 162, 225)
    COLOR_PLAYER_THRUST = (255, 255, 255)
    COLOR_OBSTACLE = (232, 77, 82)
    COLOR_COIN = (242, 212, 80)
    COLOR_PLATFORM = (90, 105, 120)
    COLOR_TEXT = (240, 240, 240)
    
    # Physics
    GRAVITY = 0.4
    JUMP_STRENGTH = -8.5
    PLAYER_SPEED = 4.0
    MAX_FALL_SPEED = 10.0

    # Game Rules
    TOTAL_STAGES = 3
    LIVES_START = 3
    STAGE_TIME_SECONDS = 60
    FPS = 30
    INVINCIBILITY_FRAMES = 60 # 2 seconds

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
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.is_on_ground = None
        self.lives = None
        self.score = None
        self.current_stage = None
        self.stage_timer = None
        self.invincibility_timer = None
        self.camera_offset_x = None
        self.max_x_reached = None

        self.platforms = []
        self.obstacles = []
        self.coins = []
        self.particles = []
        self.stars = []
        self.stage_end_x = 0
        self.game_over = False
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.LIVES_START
        self.current_stage = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(100, 200)
        self.player_vel = pygame.Vector2(0, 0)
        self.invincibility_timer = 0
        self.camera_offset_x = 0
        self.max_x_reached = self.player_pos.x

        self._create_stars()
        self._setup_stage(self.current_stage)
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self, stage_index):
        self.platforms.clear()
        self.obstacles.clear()
        self.coins.clear()
        self.particles.clear()
        
        self.stage_timer = self.STAGE_TIME_SECONDS * self.FPS
        self.player_pos.x = 100
        self.player_pos.y = 200
        self.player_vel.x = 0
        self.player_vel.y = 0
        self.is_on_ground = False
        self.max_x_reached = self.player_pos.x

        # Procedural Stage Generation
        difficulty_multiplier = 1.0 + (stage_index * 0.1)
        current_x = -200
        current_y = 350
        
        # Initial platform
        self.platforms.append(pygame.Rect(current_x, current_y, 400, 50))

        for i in range(20 + stage_index * 5):
            gap = self.np_random.integers(80, 150)
            current_x += self.platforms[-1].width + gap
            
            width = self.np_random.integers(150, 400)
            height_change = self.np_random.integers(-80, 80)
            current_y = np.clip(current_y + height_change, 150, self.SCREEN_HEIGHT - 50)
            
            platform_rect = pygame.Rect(current_x, current_y, width, 50)
            self.platforms.append(platform_rect)
            
            # Add coins
            if self.np_random.random() < 0.8:
                num_coins = self.np_random.integers(2, 5)
                for j in range(num_coins):
                    coin_x = platform_rect.x + (platform_rect.width / (num_coins + 1)) * (j + 1)
                    coin_y = platform_rect.y - 40 - self.np_random.integers(0, 30)
                    self.coins.append({"pos": pygame.Vector2(coin_x, coin_y), "anim_offset": self.np_random.random() * 100})

            # Add obstacles
            if self.np_random.random() < 0.4 + stage_index * 0.15:
                obstacle_type = self.np_random.choice(['patrol_h', 'patrol_v', 'static'])
                ox = platform_rect.centerx
                oy = platform_rect.top - 20
                
                if obstacle_type == 'patrol_h':
                    self.obstacles.append({"pos": pygame.Vector2(ox, oy), "type": "patrol_h", "start_x": platform_rect.left, "end_x": platform_rect.right, "speed": self.np_random.uniform(1.0, 2.0) * difficulty_multiplier})
                elif obstacle_type == 'patrol_v':
                     self.obstacles.append({"pos": pygame.Vector2(ox, oy - 50), "type": "patrol_v", "start_y": oy - 100, "end_y": oy, "speed": self.np_random.uniform(1.0, 2.0) * difficulty_multiplier})
                else: # static
                    self.obstacles.append({"pos": pygame.Vector2(ox, oy), "type": "static"})
        
        self.stage_end_x = self.platforms[-1].right + 200

    def _create_stars(self):
        self.stars.clear()
        for _ in range(150):
            self.stars.append({
                "pos": pygame.Vector2(self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                "depth": self.np_random.uniform(0.1, 0.8) # Slower stars are further away
            })

    def step(self, action):
        movement = action[0]
        reward = -0.01  # Cost of living
        terminated = False
        
        # --- Handle Input ---
        if movement == 1 and self.is_on_ground:  # Jump
            self.player_vel.y = self.JUMP_STRENGTH
            self.is_on_ground = False
            # sfx: jump
            self._create_particles(self.player_pos + pygame.Vector2(0, 15), 10, (200, 200, 255), 1, 3, y_dir=(0, 1))

        player_target_vel_x = 0
        if movement == 3:  # Left
            player_target_vel_x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            player_target_vel_x = self.PLAYER_SPEED
        
        # Smooth horizontal movement
        self.player_vel.x += (player_target_vel_x - self.player_vel.x) * 0.3

        # --- Update Physics & State ---
        self.steps += 1
        self.stage_timer -= 1
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)

        # Update position
        self.player_pos += self.player_vel
        
        # Reward for moving right
        progress = self.player_pos.x - self.max_x_reached
        if progress > 0:
            reward += progress * 0.1 # Reward for progress
            self.max_x_reached = self.player_pos.x

        # --- Collision Detection ---
        player_rect = self._get_player_rect()
        on_ground_this_frame = False

        # Platform collision
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Check if landing on top
                if self.player_vel.y > 0 and player_rect.bottom - self.player_vel.y <= plat.top + 1:
                    self.player_pos.y = plat.top - player_rect.height / 2
                    self.player_vel.y = 0
                    if not self.is_on_ground: # First frame of landing
                        # sfx: land
                        self._create_particles(pygame.Vector2(player_rect.midbottom), 5, self.COLOR_PLATFORM, 0.5, 2, (-1, 1), (-0.5, 0))
                    on_ground_this_frame = True
                # Hitting from below
                elif self.player_vel.y < 0 and player_rect.top - self.player_vel.y >= plat.bottom - 1:
                    self.player_pos.y = plat.bottom + player_rect.height / 2
                    self.player_vel.y = 0
                # Side collision
                else:
                    if self.player_vel.x > 0:
                        self.player_pos.x = plat.left - player_rect.width / 2
                    elif self.player_vel.x < 0:
                        self.player_pos.x = plat.right + player_rect.width / 2
                    self.player_vel.x = 0
        
        self.is_on_ground = on_ground_this_frame

        # Obstacle collision
        if self.invincibility_timer == 0:
            for obs in self.obstacles:
                obs_rect = pygame.Rect(obs['pos'].x - 10, obs['pos'].y - 10, 20, 20)
                if player_rect.colliderect(obs_rect):
                    self.lives -= 1
                    reward -= 5
                    self.invincibility_timer = self.INVINCIBILITY_FRAMES
                    # sfx: player_hit
                    self._create_particles(self.player_pos, 20, self.COLOR_OBSTACLE, 2, 5)
                    if self.lives <= 0:
                        terminated = True
                    break
        
        # Coin collection
        for coin in self.coins[:]:
            if player_rect.collidepoint(coin['pos']):
                self.coins.remove(coin)
                self.score += 10
                reward += 10
                # sfx: coin_collect
                self._create_particles(coin['pos'], 15, self.COLOR_COIN, 1, 3)
        
        # --- Update Dynamic Elements ---
        self._update_particles()
        self._update_obstacles()
        self._update_camera()

        # --- Check Termination Conditions ---
        if self.player_pos.y > self.SCREEN_HEIGHT + 100: # Fell off world
            self.lives = 0
            terminated = True
        
        if self.stage_timer <= 0:
            terminated = True
        
        if terminated:
            self.game_over = True
        
        # Check Stage Completion
        if not terminated and self.player_pos.x >= self.stage_end_x:
            self.score += 50
            reward += 50
            self.current_stage += 1
            if self.current_stage >= self.TOTAL_STAGES:
                terminated = True
                self.game_over = True
                reward += 100 # Bonus for winning
                # sfx: game_win
            else:
                self._setup_stage(self.current_stage)
                # sfx: stage_complete

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 15, 20, 30)

    def _update_obstacles(self):
        t = self.steps
        for obs in self.obstacles:
            if obs['type'] == 'patrol_h':
                center = (obs['start_x'] + obs['end_x']) / 2
                width = (obs['end_x'] - obs['start_x']) / 2
                obs['pos'].x = center + math.sin(t * 0.02 * obs['speed']) * width
            elif obs['type'] == 'patrol_v':
                center = (obs['start_y'] + obs['end_y']) / 2
                height = (obs['end_y'] - obs['start_y']) / 2
                obs['pos'].y = center + math.sin(t * 0.02 * obs['speed']) * height

    def _update_camera(self):
        target_cam_x = self.player_pos.x - self.SCREEN_WIDTH / 3
        self.camera_offset_x += (target_cam_x - self.camera_offset_x) * 0.1

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, min_speed, max_speed, x_dir=(-1, 1), y_dir=(-1, 1)):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(
                    self.np_random.uniform(x_dir[0], x_dir[1]),
                    self.np_random.uniform(y_dir[0], y_dir[1])
                ).normalize() * self.np_random.uniform(min_speed, max_speed),
                "lifespan": self.np_random.integers(10, 25),
                "color": color,
                "size": self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x = int(self.camera_offset_x)
        cam_y = 0 # No vertical scroll

        # Render Stars (parallax)
        for star in self.stars:
            x = (star['pos'].x - cam_x * star['depth']) % self.SCREEN_WIDTH
            y = star['pos'].y
            size = int(star['depth'] * 2)
            brightness = int(100 + star['depth'] * 155)
            pygame.draw.rect(self.screen, (brightness, brightness, brightness), (x, y, size, size))

        # Render Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-cam_x, -cam_y))

        # Render Obstacles
        for obs in self.obstacles:
            pos = obs['pos'] - pygame.Vector2(cam_x, cam_y)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 10, self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 10, self.COLOR_OBSTACLE)

        # Render Coins
        for coin in self.coins:
            t = self.steps + coin['anim_offset']
            size = 8 + math.sin(t * 0.1) * 2
            pos = coin['pos'] - pygame.Vector2(cam_x, cam_y)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(size), self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(size), self.COLOR_COIN)
            # Sparkle
            if self.np_random.random() < 0.1:
                sparkle_angle = self.np_random.random() * 2 * math.pi
                sparkle_dist = size + 2
                start_pos = pos + pygame.Vector2(math.cos(sparkle_angle), math.sin(sparkle_angle)) * sparkle_dist
                end_pos = start_pos + pygame.Vector2(math.cos(sparkle_angle), math.sin(sparkle_angle)) * 3
                pygame.draw.aaline(self.screen, (255, 255, 255), start_pos, end_pos)

        # Render Particles
        for p in self.particles:
            pos = p['pos'] - pygame.Vector2(cam_x, cam_y)
            # alpha = max(0, p['lifespan'] * 10)
            # color = (*p['color'], alpha)
            pygame.draw.circle(self.screen, p['color'], pos, int(p['size'] * (p['lifespan']/25.0)))

        # Render Player
        if self.invincibility_timer == 0 or (self.invincibility_timer > 0 and self.steps % 10 < 5):
            player_screen_pos = self.player_pos - pygame.Vector2(cam_x, cam_y)
            p_rect = self._get_player_rect()
            p_rect.center = player_screen_pos

            # Body
            body_points = [
                (p_rect.centerx, p_rect.top),
                (p_rect.right, p_rect.centery),
                (p_rect.centerx, p_rect.bottom),
                (p_rect.left, p_rect.centery)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, body_points)
            
            # Cockpit
            pygame.draw.circle(self.screen, (200, 220, 255), p_rect.center, 5)

            # Thruster
            if self.player_vel.y < -0.5:
                thrust_y = p_rect.bottom
                for i in range(3):
                    offset_x = self.np_random.uniform(-5, 5)
                    length = self.np_random.uniform(5, 15)
                    pygame.draw.line(self.screen, self.COLOR_PLAYER_THRUST, (p_rect.centerx + offset_x, thrust_y), (p_rect.centerx + offset_x, thrust_y + length), 2)
    
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            ship_icon_points = [
                (self.SCREEN_WIDTH - 25 - i * 25, 15),
                (self.SCREEN_WIDTH - 15 - i * 25, 20),
                (self.SCREEN_WIDTH - 25 - i * 25, 25),
                (self.SCREEN_WIDTH - 35 - i * 25, 20)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, ship_icon_points)

        # Timer
        time_left = max(0, self.stage_timer // self.FPS)
        time_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_OBSTACLE
        time_text = self.font_small.render(f"TIME: {time_left}", True, time_color)
        time_rect = time_text.get_rect(centerx=self.SCREEN_WIDTH / 2, y=10)
        self.screen.blit(time_text, time_rect)

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.current_stage + 1}/{self.TOTAL_STAGES}", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(right=self.SCREEN_WIDTH - 120, y=10)
        self.screen.blit(stage_text, stage_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.lives <= 0 or self.stage_timer <= 0:
                end_text_str = "GAME OVER"
            else:
                end_text_str = "YOU WIN!"
            
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, end_rect)

            final_score_text = self.font_small.render(f"FINAL SCORE: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.current_stage + 1,
            "time_left": max(0, self.stage_timer // self.FPS)
        }

    def close(self):
        pygame.quit()
        

# Example of how to run the environment
if __name__ == '__main__':
    # Un-comment the line below to run with display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # Use arrow keys for movement, up to jump
    
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Hopping Spaceship")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # Map keyboard inputs to the discrete action space
        action = np.array([0, 0, 0]) # Default no-op
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1 # Jump
        elif keys[pygame.K_DOWN]:
            action[0] = 2 # Not used, but maps to a discrete action
        if keys[pygame.K_LEFT]:
            action[0] = 3 # Move Left
        elif keys[pygame.K_RIGHT]:
            action[0] = 4 # Move Right
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Convert observation back to a Pygame surface for display
            display_obs = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(display_obs)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

    env.close()