import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:57:28.255064
# Source Brief: brief_02475.md
# Brief Index: 2475
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a ball through a maze by applying impulses. Maintain momentum, avoid gravity wells, and reach the exit before time runs out."
    )
    user_guide = (
        "Controls: Press space to launch. Use ↑↓←→ arrow keys to apply thrust in the corresponding direction."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30.0
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (220, 220, 255)
    COLOR_BALL = (0, 191, 255)
    COLOR_BALL_GLOW = (0, 191, 255, 60)
    COLOR_EXIT = (50, 255, 50)
    COLOR_EXIT_GLOW = (50, 255, 50, 60)
    COLOR_WELL = (255, 50, 50)
    COLOR_WELL_GLOW = (255, 50, 50, 80)
    COLOR_TRAIL = (0, 120, 200)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (30, 30, 30)

    # Game parameters
    BALL_RADIUS = 8
    MAX_LEVELS = 5
    LEVEL_TIME = 60.0
    IMPULSE_STRENGTH = 200.0
    INITIAL_LAUNCH_SPEED = 250.0
    MAX_SPEED = 400.0
    WELL_MOMENTUM_DRAIN = 0.95 # Retain 95% of velocity
    WELL_RADIUS = 15
    TRAIL_LENGTH = 15

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 30, bold=True)
        
        self.dt = 1.0 / self.FPS
        
        # Initialize state variables
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.momentum = 0.0
        self.trail = deque(maxlen=self.TRAIL_LENGTH)
        
        self.level = 1
        self.total_score = 0
        self.timer = self.LEVEL_TIME
        self.launched = False
        
        self.walls = []
        self.gravity_wells = []
        self.exit_pos = pygame.math.Vector2(0, 0)
        
        self.steps = 0
        self.terminated = False
        self.reward_this_step = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.total_score = 0
        self.level = 1
        self.terminated = False
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, self.terminated, False, info

        self.steps += 1
        self.reward_this_step = 0.0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- GAME LOGIC ---
        self._handle_input(movement, space_held)
        self._update_physics()
        self._check_collisions_and_events()
        
        # Update timer
        self.timer -= self.dt
        
        # --- REWARD CALCULATION ---
        # Continuous reward for maintaining momentum
        if self.momentum > 50:
            self.reward_this_step += 0.01

        # --- TERMINATION CHECK ---
        self._check_termination_conditions()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return (
            observation,
            self.reward_this_step,
            self.terminated,
            False,
            info
        )

    def _setup_level(self):
        """Initializes the state for the current level."""
        self.timer = self.LEVEL_TIME
        self.launched = False
        self.trail.clear()

        self.ball_vel = pygame.math.Vector2(0, 0)
        self.momentum = 0.0

        self.walls, self.gravity_wells, self.ball_pos, self.exit_pos = self._generate_level_layout()

    def _generate_level_layout(self):
        """Creates the maze, wells, start, and end points for the current level."""
        walls = [
            # Outer boundary
            pygame.Rect(0, 0, self.SCREEN_WIDTH, 10),
            pygame.Rect(0, self.SCREEN_HEIGHT - 10, self.SCREEN_WIDTH, 10),
            pygame.Rect(0, 0, 10, self.SCREEN_HEIGHT),
            pygame.Rect(self.SCREEN_WIDTH - 10, 0, 10, self.SCREEN_HEIGHT),
        ]
        wells = []
        
        # Ball always starts at the bottom middle
        start_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40)
        
        if self.level == 1:
            walls.append(pygame.Rect(100, 150, 440, 10))
            wells.append(pygame.math.Vector2(320, 250))
            exit_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, 50)
        elif self.level == 2:
            walls.append(pygame.Rect(10, 280, 400, 10))
            walls.append(pygame.Rect(230, 10, 400, 10))
            wells.append(pygame.math.Vector2(480, 340))
            wells.append(pygame.math.Vector2(150, 80))
            exit_pos = pygame.math.Vector2(50, 50)
        elif self.level == 3:
            walls.append(pygame.Rect(150, 10, 10, 250))
            walls.append(pygame.Rect(480, 150, 10, 240))
            walls.append(pygame.Rect(250, 180, 150, 10))
            wells.append(pygame.math.Vector2(80, 250))
            wells.append(pygame.math.Vector2(320, 100))
            wells.append(pygame.math.Vector2(560, 100))
            exit_pos = pygame.math.Vector2(320, 350)
        elif self.level == 4:
            walls.append(pygame.Rect(0, 100, 250, 10))
            walls.append(pygame.Rect(self.SCREEN_WIDTH - 250, 100, 250, 10))
            walls.append(pygame.Rect(0, 300, 250, 10))
            walls.append(pygame.Rect(self.SCREEN_WIDTH - 250, 300, 250, 10))
            walls.append(pygame.Rect(315, 100, 10, 210))
            wells.append(pygame.math.Vector2(150, 50))
            wells.append(pygame.math.Vector2(490, 50))
            wells.append(pygame.math.Vector2(150, 350))
            wells.append(pygame.math.Vector2(490, 350))
            exit_pos = pygame.math.Vector2(320, 200)
        else: # Level 5
            walls.append(pygame.Rect(100, 70, 10, 260))
            walls.append(pygame.Rect(530, 70, 10, 260))
            walls.append(pygame.Rect(220, 10, 10, 150))
            walls.append(pygame.Rect(410, 240, 10, 150))
            walls.append(pygame.Rect(220, 240, 200, 10))
            wells.append(pygame.math.Vector2(50, 50))
            wells.append(pygame.math.Vector2(590, 350))
            wells.append(pygame.math.Vector2(320, 150))
            wells.append(pygame.math.Vector2(160, 350))
            wells.append(pygame.math.Vector2(480, 50))
            exit_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, 50)
            
        return walls, wells, start_pos, exit_pos

    def _handle_input(self, movement, space_held):
        if not self.launched and space_held:
            self.launched = True
            # SFX: Ball Launch
            self.ball_vel = pygame.math.Vector2(0, -self.INITIAL_LAUNCH_SPEED)
        
        if self.launched:
            impulse = pygame.math.Vector2(0, 0)
            if movement == 1: impulse.y = -1 # Up
            elif movement == 2: impulse.y = 1  # Down
            elif movement == 3: impulse.x = -1 # Left
            elif movement == 4: impulse.x = 1  # Right
            
            if impulse.length_squared() > 0:
                impulse.scale_to_length(self.IMPULSE_STRENGTH * self.dt)
                self.ball_vel += impulse
                # SFX: Thruster sound

    def _update_physics(self):
        if not self.launched:
            return
            
        # Cap velocity
        if self.ball_vel.length() > self.MAX_SPEED:
            self.ball_vel.scale_to_length(self.MAX_SPEED)
        
        # Update momentum (0-100 scale)
        self.momentum = (self.ball_vel.length() / self.MAX_SPEED) * 100
        
        # Store trail position
        if self.steps % 2 == 0: # Add to trail every other frame
            self.trail.append(pygame.math.Vector2(self.ball_pos))

        # --- Axis-separated collision response ---
        ball_rect = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Move X
        self.ball_pos.x += self.ball_vel.x * self.dt
        ball_rect.center = self.ball_pos
        for wall in self.walls:
            if ball_rect.colliderect(wall):
                # SFX: Wall bounce
                if self.ball_vel.x > 0: ball_rect.right = wall.left
                elif self.ball_vel.x < 0: ball_rect.left = wall.right
                self.ball_vel.x *= -1
                self.ball_pos.x = ball_rect.centerx
        
        # Move Y
        self.ball_pos.y += self.ball_vel.y * self.dt
        ball_rect.center = self.ball_pos
        for wall in self.walls:
            if ball_rect.colliderect(wall):
                # SFX: Wall bounce
                if self.ball_vel.y > 0: ball_rect.bottom = wall.top
                elif self.ball_vel.y < 0: ball_rect.top = wall.bottom
                self.ball_vel.y *= -1
                self.ball_pos.y = ball_rect.centery

    def _check_collisions_and_events(self):
        if not self.launched:
            return

        # Gravity wells
        for well_pos in self.gravity_wells:
            if self.ball_pos.distance_to(well_pos) < self.BALL_RADIUS + self.WELL_RADIUS:
                # SFX: Energy drain
                self.ball_vel *= self.WELL_MOMENTUM_DRAIN
                self.reward_this_step -= 1.0
                # Move ball slightly away to prevent repeated fast collisions
                direction_away = (self.ball_pos - well_pos).normalize()
                self.ball_pos += direction_away * 2
        
        # Exit
        if self.ball_pos.distance_to(self.exit_pos) < self.BALL_RADIUS + 10: # 10 is half exit size
            # SFX: Level complete
            self.reward_this_step += 5.0
            self.total_score += 5
            
            if self.level < self.MAX_LEVELS:
                self.level += 1
                self._setup_level()
            else:
                # Game won!
                self.reward_this_step += 100.0
                self.total_score += 100
                self.terminated = True

    def _check_termination_conditions(self):
        if self.terminated: # Already terminated by winning
            return
            
        if self.timer <= 0:
            # SFX: Time out failure
            self.reward_this_step -= 10.0
            self.total_score -= 10
            self.terminated = True
        
        if self.launched and self.momentum < 1.0:
            # SFX: Momentum loss failure
            self.reward_this_step -= 50.0
            self.total_score -= 50
            self.terminated = True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Render gravity wells with pulsating glow
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
        well_glow_radius = self.WELL_RADIUS + 5 + pulse * 5
        for well_pos in self.gravity_wells:
            pygame.gfxdraw.filled_circle(
                self.screen, int(well_pos.x), int(well_pos.y),
                int(well_glow_radius), self.COLOR_WELL_GLOW
            )
            pygame.gfxdraw.filled_circle(
                self.screen, int(well_pos.x), int(well_pos.y),
                self.WELL_RADIUS, self.COLOR_WELL
            )

        # Render exit with glow
        exit_rect = pygame.Rect(self.exit_pos.x - 10, self.exit_pos.y - 10, 20, 20)
        exit_glow_rect = exit_rect.inflate(10 + pulse * 8, 10 + pulse * 8)
        
        # Custom glow drawing
        s = pygame.Surface(exit_glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_EXIT_GLOW, s.get_rect(), border_radius=8)
        self.screen.blit(s, exit_glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=4)
        
        # Render trail
        for i, pos in enumerate(self.trail):
            alpha = int(255 * (i / self.TRAIL_LENGTH))
            radius = int(self.BALL_RADIUS * 0.5 * (i / self.TRAIL_LENGTH))
            if radius > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(pos.x), int(pos.y), radius,
                    (*self.COLOR_TRAIL, alpha)
                )

        # Render ball with glow
        ball_glow_radius = self.BALL_RADIUS + 3 + pulse * 3
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.ball_pos.x), int(self.ball_pos.y),
            int(ball_glow_radius), self.COLOR_BALL_GLOW
        )
        pygame.gfxdraw.filled_circle(
            self.screen, int(self.ball_pos.x), int(self.ball_pos.y),
            self.BALL_RADIUS, self.COLOR_BALL
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.ball_pos.x), int(self.ball_pos.y),
            self.BALL_RADIUS, self.COLOR_BALL
        )

    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, pos, font, color, shadow_color):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        # Level
        level_text = f"LEVEL: {self.level}/{self.MAX_LEVELS}"
        draw_text(level_text, (20, 20), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Timer
        timer_text = f"TIME: {max(0, self.timer):.1f}"
        timer_size = self.font_ui.size(timer_text)
        draw_text(timer_text, (self.SCREEN_WIDTH - timer_size[0] - 20, 20), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Momentum
        momentum_text = f"MOMENTUM: {int(self.momentum)}%"
        momentum_size = self.font_ui.size(momentum_text)
        draw_text(momentum_text, (self.SCREEN_WIDTH - momentum_size[0] - 20, 45), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Score
        score_text = f"SCORE: {self.total_score}"
        draw_text(score_text, (20, 45), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Game Over / Win message
        if self.terminated:
            msg = "GAME OVER"
            try:
                # This check might be fragile if exit condition changes
                if self.level >= self.MAX_LEVELS and self.ball_pos.distance_to(self.exit_pos) < self.BALL_RADIUS + 10:
                    msg = "VICTORY!"
            except ValueError: # distance_to empty vector
                pass

            msg_surf = self.font_title.render(msg, True, self.COLOR_TEXT)
            shadow_surf = self.font_title.render(msg, True, self.COLOR_TEXT_SHADOW)
            pos = (
                self.SCREEN_WIDTH/2 - msg_surf.get_width()/2,
                self.SCREEN_HEIGHT/2 - msg_surf.get_height()/2
            )
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(msg_surf, pos)

    def _get_info(self):
        return {
            "score": self.total_score,
            "steps": self.steps,
            "level": self.level,
            "timer": self.timer,
            "momentum": self.momentum,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Switch to a visible driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Momentum")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Player Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Final Info: {info}")
            # In a real scenario, you'd reset here. For manual play, we wait for 'R'.
        
        # --- Rendering ---
        # The observation is already a rendered frame. We just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()