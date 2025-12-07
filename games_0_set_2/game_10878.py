import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:12:24.551429
# Source Brief: brief_00878.md
# Brief Index: 878
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a size-shifting bouncing ball.
    The goal is to hit a target zone after achieving a minimum number of bounces
    within a time limit. The game prioritizes visual quality and "game feel".

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - actions[1]: Size control (0=small, 1=large)
    - actions[2]: Unused (Shift key)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +0.1 for each bounce off the floor.
    - +50 for reaching the goal with >= 15 bounces.
    - -0.06 per step (time penalty).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a size-shifting ball and bounce it to gain momentum. "
        "Hit the target zone after enough bounces to win before time runs out."
    )
    user_guide = (
        "Controls: ←→ to move the ball. Hold space to make the ball larger and heavier, which affects its bounce."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 1800  # 30 seconds at 60 FPS

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_BALL = (255, 80, 80)
    COLOR_BALL_OUTLINE = (255, 120, 120)
    COLOR_GOAL = (80, 255, 120)
    COLOR_FLOOR = (180, 180, 190)
    COLOR_TEXT = (230, 230, 240)
    COLOR_PARTICLE = (255, 255, 255)

    # Physics
    GRAVITY = 0.18
    BALL_ACCEL = 0.35
    FLOOR_Y = 380
    MIN_BOUNCES_TO_WIN = 15

    # Ball Properties
    SMALL_RADIUS, LARGE_RADIUS = 10, 22
    SMALL_RESTITUTION, LARGE_RESTITUTION = 0.92, 0.75 # Bounce height
    SMALL_DRAG, LARGE_DRAG = 0.999, 0.985 # Air resistance

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 48, bold=True)

        # Game State Variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""

        self.ball_pos = [0.0, 0.0]
        self.ball_vel = [0.0, 0.0]
        self.is_large = False
        self.bounce_count = 0
        self.particles = []

        self.goal_rect = pygame.Rect(self.WIDTH - 80, self.HEIGHT - 120, 60, 100)
        
        # This will be called once to ensure everything is set up for the first reset
        # self.reset() # reset is called by the wrapper/runner
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""

        # Reset ball at a random-ish starting position
        self.ball_pos = [self.np_random.uniform(50, 150), self.np_random.uniform(50, 150)]
        self.ball_vel = [self.np_random.uniform(1, 3), 0.0]
        self.is_large = False
        self.bounce_count = 0
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return (
                self._get_observation(),
                0.0,
                True,
                False,
                self._get_info()
            )

        # --- Unpack Action ---
        movement, size_action, _ = action
        self.is_large = (size_action == 1)

        # --- Game Logic Update ---
        self.steps += 1
        reward = -0.06 # Time penalty

        # 1. Handle Player Input
        if movement == 3:  # Left
            self.ball_vel[0] -= self.BALL_ACCEL
        elif movement == 4:  # Right
            self.ball_vel[0] += self.BALL_ACCEL

        # 2. Apply Physics
        restitution = self.LARGE_RESTITUTION if self.is_large else self.SMALL_RESTITUTION
        drag = self.LARGE_DRAG if self.is_large else self.SMALL_DRAG
        radius = self.LARGE_RADIUS if self.is_large else self.SMALL_RADIUS

        self.ball_vel[1] += self.GRAVITY
        self.ball_vel[0] *= drag
        self.ball_vel[1] *= drag
        
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # 3. Handle Collisions
        # Floor
        if self.ball_pos[1] + radius > self.FLOOR_Y:
            self.ball_pos[1] = self.FLOOR_Y - radius
            self.ball_vel[1] *= -restitution
            # Dampen horizontal velocity slightly on bounce
            self.ball_vel[0] *= 0.98
            self.bounce_count += 1
            reward += 0.1
            self._create_particles(self.ball_pos[0], self.FLOOR_Y)
            # Sound: Boing!

        # Walls
        if self.ball_pos[0] - radius < 0:
            self.ball_pos[0] = radius
            self.ball_vel[0] *= -0.9
        elif self.ball_pos[0] + radius > self.WIDTH:
            self.ball_pos[0] = self.WIDTH - radius
            self.ball_vel[0] *= -0.9

        # Ceiling
        if self.ball_pos[1] - radius < 0:
            self.ball_pos[1] = radius
            self.ball_vel[1] *= -0.9

        # 4. Check Termination Conditions
        terminated = False
        ball_rect = pygame.Rect(self.ball_pos[0] - radius, self.ball_pos[1] - radius, radius * 2, radius * 2)
        
        has_won = self.goal_rect.colliderect(ball_rect) and self.bounce_count >= self.MIN_BOUNCES_TO_WIN
        has_timed_out = self.steps >= self.MAX_STEPS

        if has_won:
            reward += 50.0
            terminated = True
            self.game_over = True
            self.win_message = "VICTORY!"
            # Sound: Win Jingle
        elif has_timed_out:
            terminated = True
            self.game_over = True
            self.win_message = "TIME'S UP"
            # Sound: Lose Buzzer

        self.score += reward
        
        truncated = False # This env doesn't truncate

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # --- Render all game elements ---
        # Clear screen
        self.screen.fill(self.COLOR_BG)

        # Update and Render Particles
        self._update_and_draw_particles()

        # Render static elements
        pygame.draw.line(self.screen, self.COLOR_FLOOR, (0, self.FLOOR_Y), (self.WIDTH, self.FLOOR_Y), 3)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal_rect)
        pygame.draw.rect(self.screen, tuple(int(c*0.8) for c in self.COLOR_GOAL), self.goal_rect, 3)


        # Render Ball
        radius = self.LARGE_RADIUS if self.is_large else self.SMALL_RADIUS
        pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        
        # Antialiased drawing for a high-quality look
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_BALL_OUTLINE)

        # Render UI
        self._render_ui()
        
        # Render Game Over Message
        if self.game_over:
            self._render_game_over_message()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bounce_count": self.bounce_count,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _render_ui(self):
        # Bounces display
        bounce_text = f"Bounces: {self.bounce_count}/{self.MIN_BOUNCES_TO_WIN}"
        bounce_surf = self.font_ui.render(bounce_text, True, self.COLOR_TEXT)
        self.screen.blit(bounce_surf, (15, 10))

        # Timer display
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"Time: {time_left:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 15, 10))

    def _render_game_over_message(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        # Message text
        msg_surf = self.font_msg.render(self.win_message, True, self.COLOR_TEXT)
        msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, x, y):
        # Sound: Particle pop
        num_particles = 15 if self.is_large else 8
        for _ in range(num_particles):
            angle = self.np_random.uniform(math.pi, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.uniform(20, 40)
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': lifetime, 'max_life': lifetime})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += self.GRAVITY * 0.2 # Particles are slightly affected by gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                # Fade out particles
                alpha = int(255 * (p['life'] / p['max_life']))
                # Create a temporary surface for alpha blending
                particle_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
                color = self.COLOR_PARTICLE
                size = int(3 * (p['life'] / p['max_life']))
                if size > 0:
                    pygame.draw.rect(particle_surf, (color[0], color[1], color[2], alpha), (0, 0, size, size))
                    self.screen.blit(particle_surf, (int(p['pos'][0]), int(p['pos'][1])))


    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == "__main__":
    # Set a non-dummy driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    pygame.display.init()
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Bouncing Ball Gym Environment")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")
    print("----------------\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                print("--- Environment Reset ---")

        # Get keyboard state for continuous actions
        keys = pygame.key.get_pressed()
        
        # Action mapping
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        
        action = [movement, space_held, 0] # Last element is unused shift

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Bounces: {info['bounce_count']}, Steps: {info['steps']}")
            # Wait for reset key
            
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()