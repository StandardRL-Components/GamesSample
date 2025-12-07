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


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓ to aim, Hold Space to launch the ball. Score 20 points to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An arcade basketball challenge. Aim your shot and launch the ball into the moving basket. You have 5 balls to score 20 points. Good luck!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    WIN_SCORE = 20
    MAX_BALLS = 5
    MAX_STEPS = 2000 # Increased to allow for more complex shots
    GRAVITY = 0.15
    LAUNCH_POWER = 8.0
    
    # --- Colors ---
    COLOR_BG = (15, 23, 42) # Dark Slate Blue
    COLOR_GRID = (30, 41, 59) # Lighter Slate
    COLOR_BALL = (255, 255, 255)
    COLOR_BASKET_RIM = (249, 115, 22) # Orange
    COLOR_BASKET_BACKBOARD = (148, 163, 184)
    COLOR_PLATFORM = (100, 116, 139)
    COLOR_AIMER = (255, 255, 255, 150)
    COLOR_SCORE = (74, 222, 128) # Green
    COLOR_BALLS_NORMAL = (255, 255, 255)
    COLOR_BALLS_LOW = (239, 68, 68) # Red

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)
        
        self.np_random = None
        self.ball = None
        self.basket = None
        self.particles = []
        
        # Call reset in __init__ to initialize the state
        # self.reset() # Removed to avoid calling reset before all attributes are set
        
        # self.validate_implementation() # This will be called after reset in a proper setup

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.balls_remaining = self.MAX_BALLS
        self.game_over = False
        
        self.launch_platform_y = self.HEIGHT - 40
        self.launch_angle = -math.pi / 4 
        self.min_angle = -math.pi * 0.85
        self.max_angle = -math.pi * 0.15

        self.ball = self._create_ball()
        self.basket = self._create_basket()
        
        self.particles.clear()
        
        self.last_ball_dist_to_basket = float('inf')
        
        return self._get_observation(), self._get_info()

    def _create_ball(self):
        ball = {
            "pos": pygame.Vector2(self.WIDTH / 2, self.launch_platform_y - 15),
            "vel": pygame.Vector2(0, 0),
            "radius": 8,
            "state": "IDLE", # "IDLE" or "IN_FLIGHT"
        }
        return ball

    def _create_basket(self):
        basket = {
            "pos": pygame.Vector2(self.WIDTH / 2, 80),
            "width": 60,
            "rim_thickness": 5,
            "backboard_height": 50,
            "backboard_width": 6,
            "base_speed": 1.5,
            "time": 0
        }
        return basket

    def step(self, action):
        reward = 0
        terminated = self.game_over
        
        if not terminated:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            self._handle_input(movement, space_held)
            self._update_basket()
            
            if self.ball["state"] == "IN_FLIGHT":
                step_reward = self._update_ball_flight()
                reward += step_reward
            
            self._update_particles()
            
            terminated, terminal_reward = self._check_termination()
            reward += terminal_reward
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if self.ball["state"] == "IDLE":
            # Adjust angle
            if movement == 1: # Up
                self.launch_angle -= 0.05
            elif movement == 2: # Down
                self.launch_angle += 0.05
            self.launch_angle = np.clip(self.launch_angle, self.min_angle, self.max_angle)

            # Launch ball
            if space_held and self.balls_remaining > 0:
                self.balls_remaining -= 1
                self.ball["state"] = "IN_FLIGHT"
                self.ball["vel"].x = self.LAUNCH_POWER * math.cos(self.launch_angle)
                self.ball["vel"].y = self.LAUNCH_POWER * math.sin(self.launch_angle)
                # SFX: Ball launch swoosh
                
                # Initialize distance for continuous reward
                basket_center = self.basket["pos"] + pygame.Vector2(self.basket["width"] / 2, 0)
                self.last_ball_dist_to_basket = self.ball["pos"].distance_to(basket_center)

    def _update_basket(self):
        speed_multiplier = 1.0 + (self.score // 5) * 0.1
        current_speed = self.basket["base_speed"] * speed_multiplier
        self.basket["time"] += 0.02 * current_speed
        
        amplitude = (self.WIDTH - self.basket["width"]) / 2 - 20
        center_x = self.WIDTH / 2
        self.basket["pos"].x = center_x + amplitude * math.sin(self.basket["time"])

    def _update_ball_flight(self):
        reward = 0
        
        # Apply physics
        self.ball["vel"].y += self.GRAVITY
        self.ball["pos"] += self.ball["vel"]
        
        # Add particle trail
        if self.steps % 2 == 0:
            self._add_particle(pygame.Vector2(self.ball["pos"]), self.ball["radius"] * 0.75, self.COLOR_BALL, 20)

        # Continuous reward for getting closer
        basket_center = self.basket["pos"] + pygame.Vector2(self.basket["width"] / 2, 0)
        dist = self.ball["pos"].distance_to(basket_center)
        if dist < self.last_ball_dist_to_basket:
            reward += 0.1
        else:
            reward -= 0.1
        self.last_ball_dist_to_basket = dist

        # Wall bounces
        if self.ball["pos"].x < self.ball["radius"] or self.ball["pos"].x > self.WIDTH - self.ball["radius"]:
            self.ball["vel"].x *= -0.9
            self.ball["pos"].x = np.clip(self.ball["pos"].x, self.ball["radius"], self.WIDTH - self.ball["radius"])
            # SFX: Soft thump
        
        # Backboard collision
        backboard_rect = self._get_backboard_rect()
        if backboard_rect.collidepoint(self.ball["pos"]) and self.ball["vel"].x > 0:
            self.ball["vel"].x *= -1.1 # Give it a little kick
            reward += 1.0
            # SFX: Backboard hit
        
        # Score check
        rim_rect = self._get_rim_rect()
        if rim_rect.collidepoint(self.ball["pos"]) and self.ball["vel"].y > 0:
            self.score += 1
            reward += 5.0
            # Visual effect for scoring
            for _ in range(30):
                self._add_particle(pygame.Vector2(self.ball["pos"]), random.uniform(2, 5), self.COLOR_BASKET_RIM, 30, True)
            self._reset_ball()
            # SFX: Score! Swish sound

        # Miss check
        if self.ball["pos"].y > self.launch_platform_y or self.ball["pos"].y < -50:
            self._reset_ball()
            # SFX: Ball hits floor
            
        return np.clip(reward, -10, 10)

    def _reset_ball(self):
        self.ball = self._create_ball()

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True, 100.0
        
        if self.balls_remaining == 0 and self.ball["state"] == "IDLE":
            return True, -100.0
            
        return False, 0.0

    def _add_particle(self, pos, radius, color, lifespan, is_burst=False):
        if is_burst:
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
        else:
            vel = pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        
        self.particles.append({
            "pos": pos,
            "vel": vel,
            "radius": radius,
            "color": color,
            "lifespan": lifespan,
            "max_lifespan": lifespan
        })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)
            
        # Draw launch platform
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (0, self.launch_platform_y, self.WIDTH, self.HEIGHT - self.launch_platform_y))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.launch_platform_y), (self.WIDTH, self.launch_platform_y), 2)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            radius = int(p["radius"] * (p["lifespan"] / p["max_lifespan"]))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), radius, (*p["color"], alpha))

        # Draw basket
        backboard_rect = self._get_backboard_rect()
        pygame.draw.rect(self.screen, self.COLOR_BASKET_BACKBOARD, backboard_rect)
        rim_rect = self._get_rim_rect()
        pygame.draw.rect(self.screen, self.COLOR_BASKET_RIM, rim_rect)

        # Draw aiming reticle
        if self.ball["state"] == "IDLE":
            self._draw_aim_arc()

        # Draw ball
        # Glow effect
        glow_radius = int(self.ball["radius"] * 1.8)
        glow_alpha = 70
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball["pos"].x), int(self.ball["pos"].y), glow_radius, (*self.COLOR_BALL, glow_alpha))
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball["pos"].x), int(self.ball["pos"].y), self.ball["radius"], self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball["pos"].x), int(self.ball["pos"].y), self.ball["radius"], self.COLOR_BALL)
    
    def _draw_aim_arc(self):
        temp_pos = pygame.Vector2(self.ball["pos"])
        temp_vel = pygame.Vector2(self.LAUNCH_POWER * math.cos(self.launch_angle), self.LAUNCH_POWER * math.sin(self.launch_angle))
        
        for i in range(20):
            temp_vel.y += self.GRAVITY
            temp_pos += temp_vel
            if i % 2 == 0:
                alpha = 200 - i * 8
                if alpha > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(temp_pos.x), int(temp_pos.y), 2, (*self.COLOR_AIMER[:3], alpha))

    def _get_rim_rect(self):
        return pygame.Rect(
            self.basket["pos"].x,
            self.basket["pos"].y,
            self.basket["width"],
            self.basket["rim_thickness"]
        )
    
    def _get_backboard_rect(self):
        return pygame.Rect(
            self.basket["pos"].x + self.basket["width"],
            self.basket["pos"].y - self.basket["backboard_height"] / 2 + self.basket["rim_thickness"] / 2,
            self.basket["backboard_width"],
            self.basket["backboard_height"]
        )

    def _render_ui(self):
        # Render score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 10))
        
        # Render balls remaining
        ball_color = self.COLOR_BALLS_NORMAL if self.balls_remaining > 1 else self.COLOR_BALLS_LOW
        for i in range(self.balls_remaining):
            x = self.WIDTH - 20 - i * 25
            pygame.gfxdraw.filled_circle(self.screen, x, 28, 8, ball_color)
            pygame.gfxdraw.aacircle(self.screen, x, 28, 8, ball_color)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = self.COLOR_SCORE
            else:
                msg = "GAME OVER"
                color = self.COLOR_BALLS_LOW
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_remaining": self.balls_remaining,
            "ball_state": self.ball["state"],
        }
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)

        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # In this mode, we want to see the screen.
    os.environ.pop("SDL_VIDEODRIVER", None)
    import pygame
    
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Use a separate display for human play
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Basket Launch")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # Start with no-op

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            # --- Human Controls to Action Mapping ---
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)

            if reward != 0:
                # print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
                pass
        
        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(60) # Run at 60 FPS

    env.close()