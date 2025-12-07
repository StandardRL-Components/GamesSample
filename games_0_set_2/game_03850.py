
# Generated: 2025-08-28T00:38:39.076153
# Source Brief: brief_03850.md
# Brief Index: 3850

        
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
        "Controls: ←→/↑↓ to adjust aim. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down puzzle game where you strategically bounce a ball to clear bricks within 10 turns."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500 # Failsafe termination

    COLOR_BG = (44, 62, 80)
    COLOR_BALL = (236, 240, 241)
    COLOR_LAUNCHER = (189, 195, 199)
    COLOR_TEXT = (236, 240, 241)
    COLOR_AIM_LINE = (93, 109, 126, 150)
    BRICK_COLORS = [(231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15), (155, 89, 182)]

    BALL_RADIUS = 7
    BALL_SPEED = 6.0
    LAUNCHER_HEIGHT = 380
    
    BRICK_ROWS = 5
    BRICK_COLS = 10
    BRICK_WIDTH = 60
    BRICK_HEIGHT = 20
    BRICK_AREA_TOP = 40
    BRICK_HPAD = 2
    BRICK_VPAD = 2

    TOTAL_TURNS = 10
    
    AIM_SPEED_FINE = 0.02
    AIM_SPEED_COARSE = 0.05
    MIN_ANGLE = math.pi / 12 # ~15 degrees
    MAX_ANGLE = math.pi * 11 / 12 # ~165 degrees

    REWARD_BRICK_HIT = 1
    REWARD_ROW_CLEAR = 10
    REWARD_WIN = 100
    REWARD_LOSE = -100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Etc...
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        self.np_random = None

        # Initialize state variables
        self.game_phase = "AIMING"
        self.launcher_pos = (self.SCREEN_WIDTH // 2, self.LAUNCHER_HEIGHT)
        self.launcher_angle = math.pi / 2
        self.ball = None
        self.bricks = []
        self.particles = []
        self.turns_left = self.TOTAL_TURNS
        self.initial_brick_count = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_over_message = ""
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        # Initialize all game state
        self.game_phase = "AIMING"
        self.launcher_angle = math.pi / 2
        self.ball = None
        self.particles = []
        self.turns_left = self.TOTAL_TURNS
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_over_message = ""

        self._generate_bricks()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_held = action[2] == 1
        
        self.steps += 1
        reward = 0
        terminated = self.game_over

        if not terminated:
            if self.game_phase == 'AIMING':
                reward_from_step, terminated_from_step = self._handle_aiming(movement, space_pressed)
                reward += reward_from_step
                terminated = terminated_from_step
            elif self.game_phase == 'BALL_IN_PLAY':
                reward_from_step, terminated_from_step = self._handle_ball_in_play()
                reward += reward_from_step
                terminated = terminated_from_step
        
        self.score += reward
        self.game_over = terminated

        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_aiming(self, movement, space_pressed):
        # Handle aiming adjustments based on brief
        if movement == 1:  # Up
            self.launcher_angle -= self.AIM_SPEED_FINE
        elif movement == 2:  # Down
            self.launcher_angle += self.AIM_SPEED_FINE
        elif movement == 3:  # Left
            self.launcher_angle -= self.AIM_SPEED_COARSE
        elif movement == 4:  # Right
            self.launcher_angle += self.AIM_SPEED_COARSE
        
        self.launcher_angle = max(self.MIN_ANGLE, min(self.MAX_ANGLE, self.launcher_angle))
        
        # Handle launch action
        if space_pressed and self.turns_left > 0:
            self.game_phase = 'BALL_IN_PLAY'
            self.turns_left -= 1
            
            start_pos = list(self.launcher_pos)
            vel_x = self.BALL_SPEED * math.cos(self.launcher_angle)
            vel_y = -self.BALL_SPEED * math.sin(self.launcher_angle)
            self.ball = {'pos': start_pos, 'vel': [vel_x, vel_y], 'radius': self.BALL_RADIUS}
            # Sound: launch_sound.play()
        
        return 0, False

    def _handle_ball_in_play(self):
        reward = 0
        terminated = False

        if self.ball:
            collision_reward = self._update_ball_and_collisions()
            reward += collision_reward
        
        self._update_particles()
        
        turn_over = False
        if self.ball and self.ball['pos'][1] > self.SCREEN_HEIGHT + self.ball['radius'] * 2:
            turn_over = True
        
        if turn_over:
            self.ball = None
            self.game_phase = 'AIMING'
            
            if not self.bricks:
                reward += self.REWARD_WIN
                terminated = True
                self.game_over_message = "YOU WIN!"
                # Sound: win_sound.play()
            elif self.turns_left <= 0:
                reward += self.REWARD_LOSE
                terminated = True
                self.game_over_message = "GAME OVER"
                # Sound: lose_sound.play()
        
        return reward, terminated
    
    def _generate_bricks(self):
        self.bricks = []
        total_brick_width = self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_HPAD) - self.BRICK_HPAD
        start_x = (self.SCREEN_WIDTH - total_brick_width) // 2

        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                x = start_x + c * (self.BRICK_WIDTH + self.BRICK_HPAD)
                y = self.BRICK_AREA_TOP + r * (self.BRICK_HEIGHT + self.BRICK_VPAD)
                rect = pygame.Rect(x, y, self.BRICK_WIDTH, self.BRICK_HEIGHT)
                color = self.BRICK_COLORS[r % len(self.BRICK_COLORS)]
                self.bricks.append({'rect': rect, 'color': color, 'row': r})
        self.initial_brick_count = len(self.bricks)

    def _update_ball_and_collisions(self):
        reward = 0
        if not self.ball: return 0

        # Update position
        self.ball['pos'][0] += self.ball['vel'][0]
        self.ball['pos'][1] += self.ball['vel'][1]
        
        pos, vel, radius = self.ball['pos'], self.ball['vel'], self.ball['radius']

        # Wall collisions
        if pos[0] <= radius:
            vel[0] *= -1; pos[0] = radius # Sound: wall_bounce.play()
        if pos[0] >= self.SCREEN_WIDTH - radius:
            vel[0] *= -1; pos[0] = self.SCREEN_WIDTH - radius # Sound: wall_bounce.play()
        if pos[1] <= radius:
            vel[1] *= -1; pos[1] = radius # Sound: wall_bounce.play()

        # Launcher collision
        launcher_rect = pygame.Rect(0, self.LAUNCHER_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.LAUNCHER_HEIGHT)
        ball_rect = pygame.Rect(pos[0] - radius, pos[1] - radius, radius * 2, radius * 2)
        if ball_rect.colliderect(launcher_rect) and vel[1] > 0:
            vel[1] *= -1
            pos[1] = self.LAUNCHER_HEIGHT - radius
            # Sound: launcher_bounce.play()

        # Brick collisions
        rows_hit = set()
        for i in range(len(self.bricks) - 1, -1, -1):
            brick_data = self.bricks[i]
            brick_rect = brick_data['rect']
            
            if ball_rect.colliderect(brick_rect):
                # Sound: brick_hit.play()
                
                # Simple bounce logic
                point_of_collision = pygame.math.Vector2(ball_rect.center)
                brick_center = pygame.math.Vector2(brick_rect.center)
                
                collision_vector = point_of_collision - brick_center
                
                if abs(collision_vector.x) / brick_rect.width > abs(collision_vector.y) / brick_rect.height:
                    vel[0] *= -1
                else:
                    vel[1] *= -1
                
                reward += self.REWARD_BRICK_HIT
                self._spawn_particles(brick_rect.center, brick_data['color'])
                rows_hit.add(brick_data['row'])
                self.bricks.pop(i)
                break # One brick per frame

        if rows_hit:
            for row_index in rows_hit:
                if all(b['row'] != row_index for b in self.bricks):
                    reward += self.REWARD_ROW_CLEAR
                    # Sound: row_clear.play()
        return reward

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.pop(i)
    
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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "turns_left": self.turns_left,
            "bricks_remaining": len(self.bricks),
        }

    def _render_game(self):
        # Bricks
        for brick in self.bricks:
            self._draw_beveled_rect(self.screen, brick['color'], brick['rect'])
        
        # Launcher
        pygame.draw.rect(self.screen, self.COLOR_LAUNCHER, (0, self.LAUNCHER_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.LAUNCHER_HEIGHT))
        
        # Aim assist
        if self.game_phase == 'AIMING':
            self._render_aim_assist()

        # Particles
        for p in self.particles:
            size = max(0, p['lifespan'] / 8)
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(size))
            
        # Ball
        if self.ball:
            pos = (int(self.ball['pos'][0]), int(self.ball['pos'][1]))
            radius = self.ball['radius']
            # Glow effect
            for i in range(radius, 0, -2):
                alpha = 80 * (1 - i / radius)
                glow_color = (*self.COLOR_BALL, alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i + 5, glow_color)

            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_BALL)

    def _render_aim_assist(self):
        # Simulate a few bounces for the aim line
        sim_pos = list(self.launcher_pos)
        sim_vel_x = self.BALL_SPEED * math.cos(self.launcher_angle)
        sim_vel_y = -self.BALL_SPEED * math.sin(self.launcher_angle)
        points = [sim_pos]
        
        for _ in range(150): # Simulate for a fixed number of steps
            sim_pos[0] += sim_vel_x
            sim_pos[1] += sim_vel_y

            if sim_pos[0] <= self.BALL_RADIUS or sim_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
                sim_vel_x *= -1
                points.append(list(sim_pos))
            if sim_pos[1] <= self.BALL_RADIUS:
                sim_vel_y *= -1
                points.append(list(sim_pos))
            if len(points) > 3: # Limit to a few bounces
                break
        points.append(list(sim_pos))
        
        if len(points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_AIM_LINE, False, points, 1)

    def _draw_beveled_rect(self, surface, color, rect):
        r, g, b = color
        light_color = (min(255, r + 40), min(255, g + 40), min(255, b + 40))
        dark_color = (max(0, r - 40), max(0, g - 40), max(0, b - 40))
        
        pygame.draw.rect(surface, color, rect)
        pygame.draw.line(surface, light_color, rect.topleft, rect.topright, 2)
        pygame.draw.line(surface, light_color, rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(surface, dark_color, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(surface, dark_color, rect.topright, rect.bottomright, 2)

    def _render_ui(self):
        # Turns
        turns_text = self.font_ui.render(f"TURNS: {self.turns_left}", True, self.COLOR_TEXT)
        self.screen.blit(turns_text, (10, 10))

        # Bricks
        bricks_text = self.font_ui.render(f"BRICKS: {len(self.bricks)}", True, self.COLOR_TEXT)
        self.screen.blit(bricks_text, (self.SCREEN_WIDTH - bricks_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            over_text = self.font_game_over.render(self.game_over_message, True, self.COLOR_TEXT)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # For Pygame display, we need a screen
    pygame.display.set_caption("Brick Breaker Gym Env")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        movement = 0 # None
        space_pressed = 0 # Released
        shift_pressed = 0 # Released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        elif keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
            
        if keys[pygame.K_SPACE]: space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            terminated = False

        action = np.array([movement, space_pressed, shift_pressed])
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        else: # Allow reset on game over
            if space_pressed:
                obs, info = env.reset()
                terminated = False
            
        # Blit the observation from the env to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        env.clock.tick(60)

    env.close()