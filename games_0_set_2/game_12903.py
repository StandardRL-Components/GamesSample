import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:37:48.711655
# Source Brief: brief_02903.md
# Brief Index: 2903
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Launch balls with custom gravity and angles to create magnetic chain reactions. "
        "Achieve a chain of 3 without hitting the walls too often to win."
    )
    user_guide = (
        "Controls: Use ↑↓ to adjust gravity and ←→ to aim. "
        "Press space to launch. Press shift to select a different ball."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000
    NUM_BALLS = 5

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_BORDER = (200, 200, 220)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SELECT = (255, 255, 255)
    COLOR_GRAV_UP = (100, 255, 100)
    COLOR_GRAV_DOWN = (255, 100, 100)
    COLOR_MAGNET = (255, 255, 0)
    COLOR_WALL_FLASH = (200, 0, 0)
    BALL_COLORS = [
        (0, 200, 255),  # Cyan
        (255, 100, 200), # Pink
        (100, 255, 100), # Green
        (255, 150, 50),  # Orange
        (200, 100, 255), # Purple
    ]

    # Physics
    BALL_RADIUS = 12
    LAUNCH_POWER = 10.0
    GRAVITY_STRENGTH = 0.005
    MIN_GRAVITY, MAX_GRAVITY = -0.2, 0.2
    ANGLE_STEP = 0.05  # Radians
    MAGNET_FORCE = 0.8
    DAMPING = 0.998 # Slight velocity decay
    MIN_VELOCITY_SQ = 0.01 # To prevent balls from stopping completely

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls = []
        self.particles = []
        self.selected_ball_idx = 0
        self.wall_hits = 0
        self.magnet_chain_hits = 0
        self.max_chain_hits = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.wall_flash_timer = 0
        
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is for debugging, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wall_hits = 0
        self.magnet_chain_hits = 0
        self.max_chain_hits = 0
        self.selected_ball_idx = 0
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.wall_flash_timer = 0

        self.balls = []
        start_x = self.WIDTH / (self.NUM_BALLS + 1)
        for i in range(self.NUM_BALLS):
            self.balls.append({
                "pos": pygame.math.Vector2(start_x * (i + 1), self.HEIGHT - 40),
                "vel": pygame.math.Vector2(0, 0),
                "gravity": 0.0,
                "angle": -math.pi / 2, # Straight up
                "color": self.BALL_COLORS[i],
                "is_launched": False,
                "is_magnetic": False
            })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Detect "press" events
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Handle Input ---
        self._handle_input(movement, space_pressed, shift_pressed)

        # --- Update Game Logic ---
        wall_hit_this_step = self._update_physics()
        magnet_collisions_this_step = self._handle_collisions()
        self._update_particles()
        
        self.max_chain_hits = max(self.max_chain_hits, self.magnet_chain_hits)

        # --- Calculate Reward ---
        if wall_hit_this_step:
            reward -= 5.0
            # sfx: wall_thud.wav
        else:
            reward += 0.1
        
        if magnet_collisions_this_step > 0:
            reward += 1.0 * magnet_collisions_this_step
            # sfx: magnet_spark.wav

        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not truncated:
            if self.max_chain_hits >= 3:
                reward += 100
                # sfx: win_jingle.wav
            elif self.wall_hits >= 3:
                reward -= 100
                # sfx: lose_buzzer.wav

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        if shift_pressed:
            self.selected_ball_idx = (self.selected_ball_idx + 1) % self.NUM_BALLS
            # sfx: switch_ball.wav
        
        ball = self.balls[self.selected_ball_idx]
        if not ball["is_launched"]:
            # Adjust gravity
            if movement == 1: # Up
                ball["gravity"] = max(self.MIN_GRAVITY, ball["gravity"] - self.GRAVITY_STRENGTH)
            elif movement == 2: # Down
                ball["gravity"] = min(self.MAX_GRAVITY, ball["gravity"] + self.GRAVITY_STRENGTH)
            
            # Adjust angle
            if movement == 3: # Left
                ball["angle"] -= self.ANGLE_STEP
            elif movement == 4: # Right
                ball["angle"] += self.ANGLE_STEP
            
            # Launch
            if space_pressed:
                ball["is_launched"] = True
                ball["vel"].x = self.LAUNCH_POWER * math.cos(ball["angle"])
                ball["vel"].y = self.LAUNCH_POWER * math.sin(ball["angle"])
                self.magnet_chain_hits = 0 # Reset chain for new launch
                # sfx: launch_ball.wav

    def _update_physics(self):
        wall_hit_this_step = False
        for ball in self.balls:
            if not ball["is_launched"]:
                continue
            
            # Apply gravity
            ball["vel"].y += ball["gravity"]
            
            # Apply magnetic force
            if ball["is_magnetic"]:
                for other in self.balls:
                    if other is ball or not other["is_launched"]:
                        continue
                    dist_vec = other["pos"] - ball["pos"]
                    dist_sq = dist_vec.length_squared()
                    if dist_sq > 1:
                        force_mag = self.MAGNET_FORCE / dist_sq
                        ball["vel"] += dist_vec.normalize() * force_mag

            # Update position
            ball["pos"] += ball["vel"]
            ball["vel"] *= self.DAMPING

            # Stop if moving too slow
            if ball["vel"].length_squared() < self.MIN_VELOCITY_SQ:
                ball["vel"].update(0,0)

            # Wall collisions
            if ball["pos"].x < self.BALL_RADIUS or ball["pos"].x > self.WIDTH - self.BALL_RADIUS:
                ball["vel"].x *= -1
                ball["pos"].x = max(self.BALL_RADIUS, min(ball["pos"].x, self.WIDTH - self.BALL_RADIUS))
                self.wall_hits += 1
                wall_hit_this_step = True
                self.wall_flash_timer = 0.2
            if ball["pos"].y < self.BALL_RADIUS or ball["pos"].y > self.HEIGHT - self.BALL_RADIUS:
                ball["vel"].y *= -1
                ball["pos"].y = max(self.BALL_RADIUS, min(ball["pos"].y, self.HEIGHT - self.BALL_RADIUS))
                self.wall_hits += 1
                wall_hit_this_step = True
                self.wall_flash_timer = 0.2
        return wall_hit_this_step

    def _handle_collisions(self):
        magnet_collisions_this_step = 0
        launched_balls = [b for b in self.balls if b["is_launched"]]
        for i in range(len(launched_balls)):
            for j in range(i + 1, len(launched_balls)):
                ball1 = launched_balls[i]
                ball2 = launched_balls[j]
                
                dist_vec = ball1["pos"] - ball2["pos"]
                dist_sq = dist_vec.length_squared()
                
                if dist_sq < (2 * self.BALL_RADIUS) ** 2 and dist_sq > 0:
                    # Resolve overlap
                    dist = math.sqrt(dist_sq)
                    overlap = (2 * self.BALL_RADIUS - dist) / 2
                    ball1["pos"] += (dist_vec / dist) * overlap
                    ball2["pos"] -= (dist_vec / dist) * overlap
                    
                    # Elastic collision
                    nx = dist_vec.x / dist
                    ny = dist_vec.y / dist
                    p = 2 * (ball1["vel"].x * nx + ball1["vel"].y * ny - ball2["vel"].x * nx - ball2["vel"].y * ny) / 2
                    
                    v1x = ball1["vel"].x - p * nx
                    v1y = ball1["vel"].y - p * ny
                    v2x = ball2["vel"].x + p * nx
                    v2y = ball2["vel"].y + p * ny
                    
                    ball1["vel"].update(v1x, v1y)
                    ball2["vel"].update(v2x, v2y)
                    
                    # sfx: ball_collide.wav

                    # Magnetic interaction
                    if ball1["is_magnetic"] and ball2["is_magnetic"]:
                        self.magnet_chain_hits += 1
                        magnet_collisions_this_step += 1
                        self._create_sparks( (ball1["pos"] + ball2["pos"]) / 2 )
                    
                    ball1["is_magnetic"] = True
                    ball2["is_magnetic"] = True

        return magnet_collisions_this_step

    def _create_sparks(self, pos):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": 1.0,
                "color": self.COLOR_MAGNET
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 0.05
            p["vel"] *= 0.95

    def _check_termination(self):
        if self.max_chain_hits >= 3:
            return True
        if self.wall_hits >= 3:
            return True
        # Truncation is handled separately
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Wall flash
        if self.wall_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * self.wall_flash_timer / 0.2)
            flash_surface.fill((*self.COLOR_WALL_FLASH, alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.wall_flash_timer -= 1 / self.FPS

        # Border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Particles
        for p in self.particles:
            alpha = max(0, int(255 * p["life"]))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2,2), 2)
            self.screen.blit(temp_surf, (int(p["pos"].x-2), int(p["pos"].y-2)))

        # Balls
        for i, ball in enumerate(self.balls):
            pos_int = (int(ball["pos"].x), int(ball["pos"].y))
            
            # Magnetic glow
            if ball["is_magnetic"]:
                glow_radius = int(self.BALL_RADIUS * 1.5)
                glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*self.COLOR_MAGNET, 60), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius))

            # Ball body
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, ball["color"])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, ball["color"])

            # Selection indicator
            if i == self.selected_ball_idx:
                 pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS + 4, self.COLOR_SELECT)

            # Pre-launch indicators
            if not ball["is_launched"]:
                # Launch angle
                end_pos_x = pos_int[0] + 30 * math.cos(ball["angle"])
                end_pos_y = pos_int[1] + 30 * math.sin(ball["angle"])
                pygame.draw.aaline(self.screen, self.COLOR_TEXT, pos_int, (end_pos_x, end_pos_y))
                
                # Gravity indicator
                grav_dir = -1 if ball['gravity'] < 0 else 1
                grav_color = self.COLOR_GRAV_UP if grav_dir == -1 else self.COLOR_GRAV_DOWN
                grav_mag = min(1.0, abs(ball['gravity']) / self.MAX_GRAVITY)
                if grav_mag > 0.05:
                    start_y = pos_int[1] - self.BALL_RADIUS - 10
                    end_y = start_y - 15 * grav_dir * grav_mag
                    pygame.draw.line(self.screen, grav_color, (pos_int[0], start_y), (pos_int[0], end_y), 2)
                    pygame.draw.polygon(self.screen, grav_color, [
                        (pos_int[0], end_y - 3 * grav_dir), 
                        (pos_int[0] - 4, end_y + 3 * grav_dir), 
                        (pos_int[0] + 4, end_y + 3 * grav_dir)
                    ])

    def _render_ui(self):
        # Wall hits
        wall_text = self.font_main.render(f"Wall Hits: {self.wall_hits} / 3", True, self.COLOR_TEXT)
        self.screen.blit(wall_text, (10, 10))

        # Magnet chain
        chain_text = self.font_main.render(f"Chain: {self.max_chain_hits} / 3", True, self.COLOR_TEXT)
        text_rect = chain_text.get_rect(bottomright=(self.WIDTH - 10, self.HEIGHT - 10))
        self.screen.blit(chain_text, text_rect)
        
        # Win/Loss message
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            msg = ""
            if self.max_chain_hits >= 3:
                msg = "VICTORY!"
            elif self.wall_hits >= 3:
                msg = "DEFEAT"
            elif self.steps >= self.MAX_STEPS:
                msg = "TIME UP"
            
            if msg:
                end_text = self.font_main.render(msg, True, self.COLOR_TEXT)
                end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
                self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wall_hits": self.wall_hits,
            "max_chain_hits": self.max_chain_hits
        }
    
    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example of how to run the environment for human play ---
    # Un-comment the line below to run with a display window
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for display
    pygame.display.set_caption("Magnetic Chain Reaction")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    running = True
    
    while running:
        # Default action
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        if not terminated and not truncated:
            keys = pygame.key.get_pressed()
            
            # Movement
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Space
            if keys[pygame.K_SPACE]:
                action[1] = 1
                
            # Shift
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1
            
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    env.close()