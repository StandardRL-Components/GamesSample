
# Generated: 2025-08-28T03:57:54.825452
# Source Brief: brief_02165.md
# Brief Index: 2165

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Hold shift for a big jump, press space for a small jump. Avoid the red obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop your spaceship through a hazardous asteroid field. Time your jumps to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Frame rate for game logic
        self.MAX_STEPS = 60 * self.FPS  # 60-second time limit
        self.LEVEL_END_X = 15000  # World coordinate to reach for victory
        self.GROUND_Y = self.HEIGHT - 40
        self.GRAVITY = 0.8
        self.SMALL_JUMP_VEL = -10
        self.BIG_JUMP_VEL = -15
        
        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GROUND = (60, 140, 70)
        self.COLOR_GROUND_DETAIL = (80, 160, 90)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (100, 200, 255)
        self.COLOR_FLAME = (255, 180, 0)
        self.COLOR_OBSTACLE = (220, 50, 50)
        self.COLOR_OBSTACLE_GLOW = (255, 100, 100)
        self.COLOR_OBSTACLE_HILIGHT = (255, 150, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (200, 200, 100)
        self.COLOR_STARS = (200, 200, 255)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel_y = 0
        self.player_on_ground = True
        self.player_squash = 1.0
        self.obstacles = []
        self.particles = []
        self.stars = None
        self.world_scroll_x = 0
        self.next_obstacle_dist = 0
        self.obstacle_speed = 0
        self.cleared_obstacles = set()
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize player
        self.player_pos = pygame.Vector2(self.WIDTH / 4, self.GROUND_Y)
        self.player_vel_y = 0
        self.player_on_ground = True
        self.player_squash = 1.0  # For squash/stretch animation

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.world_scroll_x = 0
        self.obstacle_speed = 4.0
        self.obstacles = []
        self.particles = []
        self.cleared_obstacles = set()
        self.next_obstacle_dist = self.WIDTH * 0.8
        
        # Initialize background stars only once
        if self.stars is None:
            self.stars = [
                {
                    "pos": pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.GROUND_Y)),
                    "depth": self.np_random.uniform(0.1, 0.6)
                } for _ in range(150)
            ]

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Actions ---
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        if self.player_on_ground:
            if shift_held:
                self.player_vel_y = self.BIG_JUMP_VEL
                self.player_on_ground = False
                self.player_squash = 1.5 # Stretch
                # sfx_big_jump()
            elif space_held:
                self.player_vel_y = self.SMALL_JUMP_VEL
                self.player_on_ground = False
                self.player_squash = 1.3 # Stretch
                # sfx_small_jump()
        
        # --- Update Game State ---
        self.steps += 1
        reward = 0.1  # Survival reward

        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_speed += 0.25

        # Update player physics
        self.player_pos.y += self.player_vel_y
        self.player_vel_y += self.GRAVITY

        # Ground collision
        if self.player_pos.y >= self.GROUND_Y:
            if not self.player_on_ground:
                self.player_squash = 0.7 # Squash
                self._create_landing_particles(10)
                # sfx_land()
            self.player_pos.y = self.GROUND_Y
            self.player_vel_y = 0
            self.player_on_ground = True
        
        self.player_squash += (1.0 - self.player_squash) * 0.1

        # Update world scroll
        self.world_scroll_x += self.obstacle_speed
        self.next_obstacle_dist -= self.obstacle_speed

        # --- Obstacle Management ---
        self._manage_obstacles()

        # --- Update Particles ---
        self._update_particles()
        
        # --- Check Termination & Calculate Reward ---
        reward, terminated = self._check_termination(reward)
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _manage_obstacles(self):
        # Spawn new obstacles
        if self.next_obstacle_dist <= 0:
            self._spawn_obstacle()
            self.next_obstacle_dist = self.np_random.uniform(self.WIDTH * 0.4, self.WIDTH * 0.7)

        # Update and remove old obstacles
        player_rect = self._get_player_rect()
        for obstacle in self.obstacles[:]:
            obstacle["pos"].x -= self.obstacle_speed
            if obstacle["pos"].x + obstacle["width"] < 0:
                self.obstacles.remove(obstacle)
            # Check for clearing an obstacle for reward
            elif obstacle not in self.cleared_obstacles and obstacle["pos"].x + obstacle["width"] < player_rect.left:
                self.score += 1
                self.cleared_obstacles.add(obstacle)
                # sfx_clear_obstacle()

    def _spawn_obstacle(self):
        width = self.np_random.uniform(40, 80)
        if self.np_random.choice([True, False]): # Top obstacle
            height = self.np_random.uniform(50, 150)
            pos_y = 0
        else: # Bottom obstacle
            height = self.np_random.uniform(50, 180)
            pos_y = self.GROUND_Y - height
        
        self.obstacles.append({
            "pos": pygame.Vector2(self.WIDTH + 50, pos_y),
            "width": width,
            "height": height
        })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"].y += 0.1
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_landing_particles(self, count):
        for _ in range(count):
            angle = self.np_random.uniform(math.pi * 1.1, math.pi * 1.9)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": self.player_pos.copy(), "vel": vel,
                "life": self.np_random.integers(15, 30),
                "size": self.np_random.uniform(1, 4)
            })

    def _check_termination(self, current_reward):
        terminated = False
        reward = current_reward

        # Check for cleared obstacles reward
        reward += len(self.cleared_obstacles.intersection(self.obstacles)) - (len(self.cleared_obstacles) - len(self.cleared_obstacles.difference(self.obstacles)))
        self.cleared_obstacles.intersection_update(self.obstacles)

        # 1. Collision
        player_rect = self._get_player_rect()
        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect(obstacle["pos"].x, obstacle["pos"].y, obstacle["width"], obstacle["height"])
            if player_rect.colliderect(obstacle_rect):
                self.game_over = terminated = True
                reward = -100
                # sfx_crash()
                return reward, terminated
        
        # 2. Timeout
        if self.steps >= self.MAX_STEPS:
            self.game_over = terminated = True
            reward = 0 # No penalty for timeout
        
        # 3. Victory
        if self.world_scroll_x >= self.LEVEL_END_X:
            self.game_over = terminated = True
            reward = 100
            self.score += 100
            # sfx_win()
        
        return reward, terminated

    def _get_player_rect(self):
        w = 20 * self.player_squash
        h = 30 / self.player_squash
        return pygame.Rect(self.player_pos.x - w/2, self.player_pos.y - h, w, h)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render Stars
        for star in self.stars:
            star_x = (star["pos"].x - self.world_scroll_x * star["depth"]) % self.WIDTH
            pygame.draw.circle(self.screen, self.COLOR_STARS, (int(star_x), int(star["pos"].y)), star["depth"] * 1.5)

        # Render Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        for i in range(0, self.WIDTH + 20, 20):
            x = (i - (self.world_scroll_x % 20))
            pygame.draw.line(self.screen, self.COLOR_GROUND_DETAIL, (x, self.GROUND_Y), (x-5, self.GROUND_Y+10), 2)

        # Render Obstacles
        for obs in self.obstacles:
            rect = pygame.Rect(int(obs["pos"].x), int(obs["pos"].y), int(obs["width"]), int(obs["height"]))
            glow_rect = rect.inflate(8, 8)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_OBSTACLE_GLOW, 50), s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_HILIGHT, rect.inflate(-4, -4), 2, border_radius=3)

        # Render Particles
        for p in self.particles:
            alpha = max(0, int(255 * (p["life"] / 30.0)))
            s = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_PARTICLE, alpha), (p["size"], p["size"]), p["size"])
            self.screen.blit(s, (int(p["pos"].x - p["size"]), int(p["pos"].y - p["size"])))

        # Render Player
        if not self.game_over:
            self._render_player()

    def _render_player(self):
        w = 20 * self.player_squash
        h = 30 / self.player_squash
        points = [
            (self.player_pos.x, self.player_pos.y - h),
            (self.player_pos.x - w/2, self.player_pos.y),
            (self.player_pos.x + w/2, self.player_pos.y)
        ]
        int_points = [(int(x), int(y)) for x, y in points]
        
        # Glow
        glow_radius = int(h * 1.2)
        glow_alpha = 100 + int(abs(1.0 - self.player_squash) * 100)
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, (*self.COLOR_PLAYER_GLOW, glow_alpha))
        self.screen.blit(s, (int(self.player_pos.x - glow_radius), int(self.player_pos.y - h/2 - glow_radius)))
        
        # Body
        pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

        # Flame
        if not self.player_on_ground:
            flame_h = abs(self.player_vel_y) * 1.5 + self.np_random.uniform(-3, 3)
            flame_w = w * 0.6
            flame_points = [
                (self.player_pos.x - flame_w/2, self.player_pos.y),
                (self.player_pos.x + flame_w/2, self.player_pos.y),
                (self.player_pos.x, self.player_pos.y + flame_h)
            ]
            int_flame_points = [(int(x), int(y)) for x, y in flame_points]
            pygame.gfxdraw.aapolygon(self.screen, int_flame_points, self.COLOR_FLAME)
            pygame.gfxdraw.filled_polygon(self.screen, int_flame_points, self.COLOR_FLAME)

    def _render_ui(self):
        # Score and Timer
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.world_scroll_x >= self.LEVEL_END_X: message = "LEVEL COMPLETE!"
            elif self.steps >= self.MAX_STEPS: message = "TIME'S UP!"
            else: message = "GAME OVER"

            msg_surf = self.font.render(message, True, self.COLOR_TEXT)
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20)))
            score_surf = self.small_font.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(score_surf, score_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "world_x": self.world_scroll_x,
            "level_end_x": self.LEVEL_END_X,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")