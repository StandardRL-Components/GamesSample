
# Generated: 2025-08-28T03:19:30.907635
# Source Brief: brief_04888.md
# Brief Index: 4888

        
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
        "Controls: ←→ to run, ↑ to jump. Collect coins and reach the end!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a robot through a procedurally generated side-scrolling world. Leap over deadly gaps and obstacles, collect coins, and race against the clock to reach the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.LEVEL_WIDTH = 15000
        self.TIME_LIMIT_SECONDS = 100
        self.MAX_STEPS = 5000

        # Player physics
        self.GRAVITY = 0.8
        self.PLAYER_JUMP_STRENGTH = -15
        self.PLAYER_MOVE_SPEED = 7
        self.PLAYER_FRICTION = 0.85

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_GLOW = (60, 160, 255, 50)
        self.COLOR_PLATFORM = (100, 110, 130)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_OBSTACLE_GLOW = (255, 80, 80, 70)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_COIN_GLOW = (255, 220, 0, 80)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_GOAL = (0, 255, 120)

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
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 30)
            self.font_game_over = pygame.font.SysFont(None, 60)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.is_on_ground = False
        self.camera_x = 0
        self.platforms = []
        self.coins = []
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.game_won = False
        self.obstacle_speed_modifier = 1.0

        # Initialize state
        # self.reset() # Called by gym.make
        
        # self.validate_implementation() # Run this if you need to debug

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = pygame.math.Vector2(100, 200)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 24, 36)
        self.is_on_ground = False
        
        self.camera_x = 0
        self._generate_level()
        
        self.particles = []
        self.steps = 0
        self.score = 0
        self.timer = self.TIME_LIMIT_SECONDS * self.FPS
        self.game_over = False
        self.game_won = False
        self.obstacle_speed_modifier = 1.0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.coins = []
        self.obstacles = []
        
        # Start platform
        self.platforms.append(pygame.Rect(0, 350, 400, 50))
        
        current_x = 400
        current_y = 350
        
        while current_x < self.LEVEL_WIDTH - 500:
            # Difficulty scaling (affects generation)
            max_gap = max(50, 150 - self.steps // 500)
            
            gap = self.np_random.integers(50, max_gap)
            current_x += gap
            
            width = self.np_random.integers(150, 400)
            
            y_change = self.np_random.integers(-80, 80)
            current_y = np.clip(current_y + y_change, 200, 370)
            
            platform_rect = pygame.Rect(current_x, current_y, width, self.HEIGHT - current_y)
            self.platforms.append(platform_rect)

            # Place coins
            if self.np_random.random() < 0.7:
                num_coins = self.np_random.integers(1, 5)
                for i in range(num_coins):
                    coin_x = platform_rect.left + (i + 1) * (platform_rect.width / (num_coins + 1))
                    coin_y = platform_rect.top - 40
                    self.coins.append(pygame.Rect(int(coin_x), int(coin_y), 12, 12))

            # Place obstacles
            if self.np_random.random() < 0.3 + (self.steps / self.MAX_STEPS) * 0.4:
                obs_x = platform_rect.left + self.np_random.integers(20, platform_rect.width - 20)
                self.obstacles.append(pygame.Rect(obs_x, platform_rect.top - 15, 15, 15))

            current_x += width

        # Goal platform
        self.platforms.append(pygame.Rect(self.LEVEL_WIDTH - 100, 350, 100, 50))

    def _create_particles(self, pos, color, count, speed_range, life_range):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(*speed_range)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = random.randint(*life_range)
            self.particles.append([pygame.math.Vector2(pos), vel, life, color])

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        old_pos_x = self.player_pos.x

        # --- Player Logic ---
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_MOVE_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_MOVE_SPEED
        else: # Friction
            self.player_vel.x *= self.PLAYER_FRICTION

        # Vertical movement (Jump)
        if movement == 1 and self.is_on_ground:
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.is_on_ground = False
            # // Jump sound
            self._create_particles(self.player_rect.midbottom, self.COLOR_PLAYER, 10, (1, 3), (10, 20))
        
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, 15) # Terminal velocity

        # --- Collision Detection and Position Update ---
        self.is_on_ground = False
        
        # Move horizontal
        self.player_pos.x += self.player_vel.x
        self.player_rect.centerx = int(self.player_pos.x)
        
        # Horizontal collision
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.x > 0:
                    self.player_rect.right = plat.left
                    self.player_vel.x = 0
                elif self.player_vel.x < 0:
                    self.player_rect.left = plat.right
                    self.player_vel.x = 0
                self.player_pos.x = self.player_rect.centerx

        # Move vertical
        self.player_pos.y += self.player_vel.y
        self.player_rect.centery = int(self.player_pos.y)
        
        # Vertical collision
        landed_this_frame = False
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel.y > 0: # Moving down
                    self.player_rect.bottom = plat.top
                    self.player_vel.y = 0
                    self.is_on_ground = True
                    landed_this_frame = True
                elif self.player_vel.y < 0: # Moving up
                    self.player_rect.top = plat.bottom
                    self.player_vel.y = 0
                self.player_pos.y = self.player_rect.centery
        
        if landed_this_frame:
            # // Land sound
             self._create_particles(self.player_rect.midbottom, self.COLOR_PLATFORM, 5, (0.5, 2), (5, 15))


        # --- Update World State ---
        self.steps += 1
        self.timer -= 1
        
        # Update particles
        self.particles = [[p[0] + p[1], p[1], p[2] - 1, p[3]] for p in self.particles if p[2] > 0]
        
        # Difficulty scaling
        if self.steps > 50 and self.steps % 250 == 0:
            self.obstacle_speed_modifier += 0.05

        # Coin collection
        collected_coins = []
        for coin in self.coins:
            if self.player_rect.colliderect(coin):
                collected_coins.append(coin)
                self.score += 10
                reward += 1
                # // Coin collect sound
                self._create_particles(coin.center, self.COLOR_COIN, 15, (1, 4), (15, 25))
        self.coins = [c for c in self.coins if c not in collected_coins]
        
        # Update camera
        self.camera_x = self.player_pos.x - self.WIDTH / 2.5
        self.camera_x = max(0, min(self.camera_x, self.LEVEL_WIDTH - self.WIDTH))

        # --- Calculate Reward ---
        # Reward for moving right
        progress = self.player_pos.x - old_pos_x
        if progress > 0:
            reward += progress * 0.1
        
        # Time penalty
        reward -= 0.001 # -0.01 per second / 30 fps ~= -0.0003. Let's make it a bit more impactful.

        # Reward for being airborne over a gap
        if not self.is_on_ground:
            is_over_gap = True
            for plat in self.platforms:
                if plat.left < self.player_pos.x < plat.right:
                    is_over_gap = False
                    break
            if is_over_gap:
                reward += 0.02
        
        # --- Check Termination ---
        terminated = False
        
        # 1. Fall off screen
        if self.player_pos.y > self.HEIGHT + 50:
            self.game_over = True
            terminated = True
            reward = -100
            # // Fall sound
        
        # 2. Obstacle collision
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs):
                self.game_over = True
                terminated = True
                reward = -100
                # // Death sound
                self._create_particles(self.player_rect.center, self.COLOR_OBSTACLE, 30, (2, 6), (20, 40))
                break

        # 3. Time runs out
        if self.timer <= 0:
            self.game_over = True
            terminated = True
            reward = -100
        
        # 4. Reached goal
        if self.player_pos.x >= self.LEVEL_WIDTH - 100:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward = 100
            # // Win sound
            self._create_particles((self.LEVEL_WIDTH-50, 325), self.COLOR_GOAL, 100, (2, 8), (40, 60))

        # 5. Max steps
        if self.steps >= self.MAX_STEPS:
            terminated = True
            # No extra penalty, let time/fall penalty handle it

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw goal
        goal_x = self.LEVEL_WIDTH - 50 - self.camera_x
        pygame.draw.rect(self.screen, self.COLOR_GOAL, (goal_x - 2, 0, 4, self.HEIGHT))
        pygame.gfxdraw.box(self.screen, (int(goal_x-10), 0, 20, self.HEIGHT), (*self.COLOR_GOAL, 30))

        # Draw platforms
        for plat in self.platforms:
            screen_rect = plat.move(-self.camera_x, 0)
            if screen_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect)

        # Draw obstacles
        for obs in self.obstacles:
            screen_rect = obs.move(-self.camera_x, 0)
            if screen_rect.colliderect(self.screen.get_rect()):
                # Draw glow
                glow_rect = screen_rect.inflate(10, 10)
                s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.circle(s, self.COLOR_OBSTACLE_GLOW, s.get_rect().center, glow_rect.width / 2)
                self.screen.blit(s, glow_rect.topleft)
                # Draw main shape (triangle)
                p1 = (screen_rect.midtop[0], screen_rect.top)
                p2 = (screen_rect.left, screen_rect.bottom)
                p3 = (screen_rect.right, screen_rect.bottom)
                pygame.draw.polygon(self.screen, self.COLOR_OBSTACLE, [p1, p2, p3])

        # Draw coins
        for coin in self.coins:
            screen_rect = coin.move(-self.camera_x, 0)
            if screen_rect.colliderect(self.screen.get_rect()):
                # Spinning animation
                spin_phase = (self.steps + coin.x) % 30 / 30
                width = int(coin.width * abs(math.cos(spin_phase * math.pi)))
                anim_rect = pygame.Rect(0, 0, max(1, width), coin.height)
                anim_rect.center = screen_rect.center
                # Draw glow
                pygame.gfxdraw.filled_circle(self.screen, int(screen_rect.centerx), int(screen_rect.centery), int(coin.width * 0.8), self.COLOR_COIN_GLOW)
                # Draw coin
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, anim_rect)

        # Draw particles
        for p in self.particles:
            pos = p[0]
            life = p[2]
            color = p[3]
            screen_pos = (int(pos.x - self.camera_x), int(pos.y))
            size = max(1, int(life / 4))
            pygame.draw.rect(self.screen, color, (*screen_pos, size, size))

        # Draw Player
        if not (self.game_over and not self.game_won):
            screen_rect = self.player_rect.move(-self.camera_x, 0)
            
            # Glow
            glow_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (30, 30), 30)
            self.screen.blit(glow_surf, (screen_rect.centerx - 30, screen_rect.centery - 30))

            # Body
            body_rect = pygame.Rect(0,0, self.player_rect.width, self.player_rect.height - 8)
            body_rect.midbottom = screen_rect.midbottom
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=4)
            
            # Head
            head_pos = (screen_rect.centerx, screen_rect.top + 6)
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, head_pos, 8)
            
            # Eye
            eye_dir = 1 if self.player_vel.x >= 0 else -1
            eye_pos = (head_pos[0] + 3 * eye_dir, head_pos[1])
            pygame.draw.circle(self.screen, self.COLOR_BG, eye_pos, 2)

            # Leg animation
            if self.is_on_ground and abs(self.player_vel.x) > 1:
                leg_phase = (self.steps // 3) % 2
                leg_y_offset = body_rect.bottom
                leg1_x = body_rect.centerx - 5 + (5 if leg_phase == 0 else -5)
                leg2_x = body_rect.centerx + 5 + (-5 if leg_phase == 0 else 5)
                pygame.draw.line(self.screen, self.COLOR_PLAYER, (body_rect.centerx-5, leg_y_offset), (leg1_x, leg_y_offset+8), 4)
                pygame.draw.line(self.screen, self.COLOR_PLAYER, (body_rect.centerx+5, leg_y_offset), (leg2_x, leg_y_offset+8), 4)
            else: # Idle/jump legs
                leg_y_offset = body_rect.bottom
                pygame.draw.line(self.screen, self.COLOR_PLAYER, (body_rect.centerx-5, leg_y_offset), (body_rect.centerx-5, leg_y_offset+8), 4)
                pygame.draw.line(self.screen, self.COLOR_PLAYER, (body_rect.centerx+5, leg_y_offset), (body_rect.centerx+5, leg_y_offset+8), 4)
                
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        time_color = self.COLOR_UI_TEXT if time_left > 10 else self.COLOR_OBSTACLE
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, time_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_GOAL if self.game_won else self.COLOR_OBSTACLE
            end_text = self.font_game_over.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            # Simple text shadow
            shadow_text = self.font_game_over.render(message, True, (0,0,0,100))
            self.screen.blit(shadow_text, text_rect.move(3,3))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.timer / self.FPS),
            "player_x": self.player_pos.x,
            "level_progress": self.player_pos.x / self.LEVEL_WIDTH
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        super().close()

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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- To validate the implementation ---
    # env.validate_implementation()
    
    # --- To play the game manually ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Procedural Platformer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        mov = 0 # no-op
        if keys[pygame.K_UP]:
            mov = 1
        elif keys[pygame.K_DOWN]:
            mov = 2
        elif keys[pygame.K_LEFT]:
            mov = 3
        elif keys[pygame.K_RIGHT]:
            mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        clock.tick(env.FPS)
        
    env.close()