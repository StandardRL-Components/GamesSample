import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# To run headless, you might need this line
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (←↑ to rotate counter-clockwise, →↓ to rotate clockwise). "
        "Press Space to hit the ball."
    )

    game_description = (
        "A vibrant, geometric mini-golf puzzle. Aim your paddle and sink the ball in all "
        "three holes as quickly as possible to maximize your score."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_TIME_SECONDS = 30
    MAX_STEPS = MAX_TIME_SECONDS * FPS + 100 # A little buffer

    # Colors
    COLOR_BG = (20, 60, 40)
    COLOR_WALL = (100, 100, 100)
    COLOR_BALL = (255, 220, 0)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_HOLE = (10, 10, 10)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)
    
    # Game Physics
    PADDLE_ROTATION_SPEED = 6  # degrees per frame
    PADDLE_LENGTH = 60
    PADDLE_WIDTH = 8
    BALL_RADIUS = 8
    BALL_HIT_POWER = 8
    BALL_FRICTION = 0.985
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_score = pygame.font.Font(None, 48)
        self.font_message = pygame.font.Font(None, 50)

        self._setup_holes()
        self.particles = []

        # This will be initialized in reset
        self.score = 0
        self.steps = 0
        self.timer = 0
        self.game_over = False
        self.current_hole_index = 0
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.paddle_angle = 0.0
        self.paddle_pivot = pygame.Vector2(0, 0)
        self.last_dist_to_hole = 0.0
        self.message = ""
        self.message_timer = 0
        
        # Call reset here to ensure the environment is initialized correctly
        # The seed is handled by the super().reset() call inside the method
        # self.reset()

    def _setup_holes(self):
        self.holes = [
            { # Hole 1: Straight shot
                "start_pos": pygame.Vector2(80, self.HEIGHT / 2),
                "hole_pos": pygame.Vector2(self.WIDTH - 80, self.HEIGHT / 2),
                "hole_radius": 15,
                "walls": []
            },
            { # Hole 2: Dogleg right
                "start_pos": pygame.Vector2(100, self.HEIGHT - 60),
                "hole_pos": pygame.Vector2(self.WIDTH - 100, 60),
                "hole_radius": 14,
                "walls": [
                    pygame.Rect(self.WIDTH / 2, self.HEIGHT / 2, 20, self.HEIGHT / 2),
                    pygame.Rect(0, self.HEIGHT / 2 - 100, self.WIDTH / 2, 20)
                ]
            },
            { # Hole 3: Obstacle course
                "start_pos": pygame.Vector2(60, 60),
                "hole_pos": pygame.Vector2(self.WIDTH - 60, self.HEIGHT - 60),
                "hole_radius": 13,
                "walls": [
                    pygame.Rect(150, 0, 20, 200),
                    pygame.Rect(self.WIDTH - 170, self.HEIGHT - 200, 20, 200),
                    pygame.Rect(self.WIDTH / 2 - 50, self.HEIGHT/2 - 10, 100, 20),
                ]
            }
        ]

    def _load_hole(self, hole_index):
        if hole_index >= len(self.holes):
            self.game_over = True
            return

        hole_data = self.holes[hole_index]
        self.current_hole_index = hole_index
        # FIX: pygame.Vector2 does not have a .copy() method.
        # Create a new vector by passing the old one to the constructor.
        self.ball_pos = pygame.Vector2(hole_data["start_pos"])
        self.ball_vel = pygame.Vector2(0, 0)
        
        # Place paddle pivot near the ball
        self.paddle_pivot = self.ball_pos + pygame.Vector2(-self.PADDLE_LENGTH / 2 - 5, 0)
        self.paddle_angle = 0
        
        current_hole_pos = self.holes[self.current_hole_index]["hole_pos"]
        self.last_dist_to_hole = self.ball_pos.distance_to(current_hole_pos)
        self._show_message(f"Hole {hole_index + 1}", 60)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.steps = 0
        self.timer = self.MAX_TIME_SECONDS * self.FPS
        self.game_over = False
        self.particles = []
        
        self._load_hole(0)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # To handle calls after termination, we can return the last state
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        reward = -0.01  # Time penalty

        # --- Action Handling ---
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        is_ball_moving = self.ball_vel.length() > 0.1

        if not is_ball_moving:
            # Rotate paddle
            if movement in [1, 3]:  # Up or Left
                self.paddle_angle += self.PADDLE_ROTATION_SPEED
            elif movement in [2, 4]:  # Down or Right
                self.paddle_angle -= self.PADDLE_ROTATION_SPEED
            self.paddle_angle %= 360

            # Hit ball
            if space_pressed:
                self.ball_vel = pygame.Vector2(self.BALL_HIT_POWER, 0).rotate(-self.paddle_angle)
                self._create_particles(self.ball_pos, 10, self.COLOR_PADDLE, 2, 4)

        # --- Physics and Game Logic Update ---
        reward += self._update_ball_state()
        self._update_particles()
        if self.message_timer > 0:
            self.message_timer -= 1

        # --- Termination Check ---
        terminated = False
        if self.game_over: # Game won
            terminated = True
            reward += 50 # Bonus for winning
            self._show_message("YOU WIN!", 120)
        elif self.timer <= 0:
            terminated = True
            reward -= 20 # Penalty for timeout
            self._show_message("TIME UP!", 120)
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_ball_state(self):
        if self.ball_vel.length() < 0.1:
            self.ball_vel = pygame.Vector2(0, 0)
            return 0
        
        reward = 0
        
        # Update position and apply friction
        self.ball_pos += self.ball_vel
        self.ball_vel *= self.BALL_FRICTION
        
        # Add trail particles
        if self.np_random.random() < 0.5:
             self._create_particles(self.ball_pos, 1, self.COLOR_BALL, 1, 2, life=10, spread=0)

        # --- Collision Detection ---
        current_hole_data = self.holes[self.current_hole_index]
        hole_pos = current_hole_data["hole_pos"]
        hole_radius = current_hole_data["hole_radius"]

        # Distance-based reward
        new_dist_to_hole = self.ball_pos.distance_to(hole_pos)
        dist_change = self.last_dist_to_hole - new_dist_to_hole
        reward += dist_change * 0.01 # Reward for getting closer
        self.last_dist_to_hole = new_dist_to_hole

        # Hole collision
        if new_dist_to_hole < hole_radius:
            reward += 10
            self.score += 100 + int(self.timer / self.FPS) # Score bonus for time left
            self._create_particles(hole_pos, 50, self.COLOR_BALL, 2, 5, life=60)
            next_hole = self.current_hole_index + 1
            if next_hole >= len(self.holes):
                self.game_over = True # Won the game
            else:
                self._load_hole(next_hole)
            return reward

        # Wall collisions
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        # Environment boundaries
        if self.ball_pos.x < self.BALL_RADIUS or self.ball_pos.x > self.WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            reward -= 5
            self._create_particles(self.ball_pos, 5, self.COLOR_WALL)
        if self.ball_pos.y < self.BALL_RADIUS or self.ball_pos.y > self.HEIGHT - self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            reward -= 5
            self._create_particles(self.ball_pos, 5, self.COLOR_WALL)

        # Hole-specific walls
        for wall in current_hole_data["walls"]:
            if wall.colliderect(ball_rect):
                overlap_x = (ball_rect.width / 2 + wall.width / 2) - abs(ball_rect.centerx - wall.centerx)
                overlap_y = (ball_rect.height / 2 + wall.height / 2) - abs(ball_rect.centery - wall.centery)

                if overlap_x < overlap_y:
                    self.ball_vel.x *= -1
                    self.ball_pos.x += np.sign(self.ball_vel.x) * overlap_x if np.sign(self.ball_vel.x) != 0 else (1 if wall.centerx < self.ball_pos.x else -1) * overlap_x
                else:
                    self.ball_vel.y *= -1
                    self.ball_pos.y += np.sign(self.ball_vel.y) * overlap_y if np.sign(self.ball_vel.y) != 0 else (1 if wall.centery < self.ball_pos.y else -1) * overlap_y
                
                reward -= 5
                self._create_particles(self.ball_pos, 5, self.COLOR_WALL)
                break
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_draw_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        hole_data = self.holes[self.current_hole_index]

        # Draw walls
        for wall in hole_data["walls"]:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Draw hole (antialiased)
        hole_pos = (int(hole_data["hole_pos"].x), int(hole_data["hole_pos"].y))
        pygame.gfxdraw.filled_circle(self.screen, hole_pos[0], hole_pos[1], hole_data["hole_radius"], self.COLOR_HOLE)
        pygame.gfxdraw.aacircle(self.screen, hole_pos[0], hole_pos[1], hole_data["hole_radius"], self.COLOR_HOLE)

        # Draw ball (antialiased with glow)
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        glow_color = (*self.COLOR_BALL, 50)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Draw paddle if ball is stationary
        if self.ball_vel.length() < 0.1:
            p1 = pygame.Vector2(self.PADDLE_LENGTH / 2, -self.PADDLE_WIDTH / 2).rotate(-self.paddle_angle) + self.paddle_pivot
            p2 = pygame.Vector2(self.PADDLE_LENGTH / 2, self.PADDLE_WIDTH / 2).rotate(-self.paddle_angle) + self.paddle_pivot
            p3 = pygame.Vector2(-self.PADDLE_LENGTH / 2, self.PADDLE_WIDTH / 2).rotate(-self.paddle_angle) + self.paddle_pivot
            p4 = pygame.Vector2(-self.PADDLE_LENGTH / 2, -self.PADDLE_WIDTH / 2).rotate(-self.paddle_angle) + self.paddle_pivot
            points = [(int(p.x), int(p.y)) for p in [p1, p2, p3, p4]]
            
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PADDLE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PADDLE)

    def _render_ui(self):
        # Semi-transparent background for UI elements
        ui_surf = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))
        self.screen.blit(ui_surf, (0, self.HEIGHT - 40))

        # Hole number
        hole_text = self.font_ui.render(f"Hole: {self.current_hole_index + 1}/{len(self.holes)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(hole_text, (10, 10))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_color = self.COLOR_UI_TEXT if time_left > 10 else (255, 100, 100)
        timer_text = self.font_ui.render(f"Time: {time_left:.1f}s", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Score
        score_text = self.font_score.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        score_pos = (self.WIDTH / 2 - score_text.get_width() / 2, self.HEIGHT - score_text.get_height())
        self.screen.blit(score_text, score_pos)

        # Messages
        if self.message_timer > 0:
            alpha = min(255, self.message_timer * 5)
            message_surf = self.font_message.render(self.message, True, self.COLOR_UI_TEXT)
            message_surf.set_alpha(alpha)
            pos = (self.WIDTH/2 - message_surf.get_width()/2, self.HEIGHT/2 - message_surf.get_height()/2)
            self.screen.blit(message_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.timer / self.FPS),
            "current_hole": self.current_hole_index + 1,
        }
        
    def _create_particles(self, pos, count, color, min_size=1, max_size=3, life=20, spread=3):
        for _ in range(count):
            angle = self.np_random.random() * 360
            speed = self.np_random.random() * spread + 1
            velocity = pygame.Vector2(speed, 0).rotate(angle)
            self.particles.append({
                # FIX: pygame.Vector2 does not have a .copy() method.
                # Create a new vector by passing the old one to the constructor.
                "pos": pygame.Vector2(pos),
                "vel": velocity,
                "life": life + self.np_random.integers(-5, 5),
                "size": self.np_random.integers(min_size, max_size + 1),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.9
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _update_and_draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p["life"] * (255 / 20))))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"].x - p["size"]), int(p["pos"].y - p["size"])))

    def _show_message(self, text, duration_frames):
        self.message = text
        self.message_timer = duration_frames

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # To run with display, comment out the os.environ line at the top of the file
    try:
        os.environ["SDL_VIDEODRIVER"]
        print("Running in headless mode. No display will be shown.")
        # Test a few steps in headless mode
        env = GameEnv()
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished.")
                obs, info = env.reset()
        env.close()
        print("Headless test completed.")
    except KeyError:
        # Normal execution with display
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Mini-Golf Puzzle")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            # --- Human Controls ---
            keys = pygame.key.get_pressed()
            move_action = 0
            if keys[pygame.K_UP] or keys[pygame.K_w]: move_action = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: move_action = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: move_action = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: move_action = 4
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [move_action, space_action, shift_action]
            
            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # --- Pygame Rendering ---
            # The observation is already a rendered frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0

            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}. Info: {info}")
                running = False 
                
            clock.tick(GameEnv.FPS)
            
        env.close()
        pygame.quit()