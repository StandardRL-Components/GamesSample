import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←/→ to aim jump. Hold Shift for a bigger jump. "
        "Use ↑ to jump straight up."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Minimalist arcade platformer. Hop between moving platforms to reach the goal "
        "at the top before time runs out."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TOTAL_TIME = 30  # seconds

        # Colors
        self.COLOR_BG_TOP = (5, 10, 50)
        self.COLOR_BG_BOTTOM = (40, 50, 110)
        self.COLOR_PLAYER = (57, 255, 20)
        self.COLOR_PLATFORM = (240, 240, 255)
        self.COLOR_GOAL = (255, 215, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (200, 200, 220)

        # Physics
        self.GRAVITY = 0.4
        self.AIR_FRICTION = 0.98
        self.JUMP_POWER_SMALL = 8.0
        self.JUMP_POWER_LARGE = 11.0
        self.JUMP_HORIZONTAL_POWER = 5.0

        # Player settings
        self.PLAYER_SIZE = 16

        # Platform settings
        self.PLATFORM_HEIGHT = 10
        self.NUM_PLATFORMS = 12

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.platforms = None
        self.goal_platform = None
        self.on_platform = None
        self.last_platform_y = None
        self.steps = None
        self.score = None
        self.time_remaining = None
        self.game_over = None
        self.game_won = None
        self.particles = None
        self.platform_speed_multiplier = None

        self.render_mode = render_mode
        self.human_screen = None
        if self.render_mode == "human":
            # This is for debugging, not part of the core env spec
            pygame.display.set_caption("Hopper")
            self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        # This will initialize all the state variables above via reset()
        # but we need a seed first.
        self.np_random = None # Will be set in reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.time_remaining = self.TOTAL_TIME * self.FPS
        self.platform_speed_multiplier = 1.0

        self._generate_platforms()

        start_platform = self.platforms[0]
        self.player_pos = [start_platform['rect'].centerx, start_platform['rect'].top - self.PLAYER_SIZE / 2]
        self.player_vel = [0, 0]
        self.on_platform = start_platform
        self.last_platform_y = start_platform['rect'].y
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.steps += 1
        self.time_remaining -= 1

        # Unpack factorized action
        movement = action[0]
        # action[1] is spacebar, not used in this logic
        shift_held = action[2] == 1

        # Update difficulty
        if self.steps == 10 * self.FPS:
            self.platform_speed_multiplier = 1.05
        elif self.steps == 20 * self.FPS:
            self.platform_speed_multiplier = 1.10

        # Update game logic
        self._update_platforms()
        event_reward = self._update_player(movement, shift_held)
        self._update_particles()

        # Calculate rewards
        reward = event_reward
        if self.on_platform:
            reward += 0.1
        else:
            reward -= 0.01

        # Check for termination
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward

        self.score += reward
        
        truncated = False # This environment does not truncate

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_platforms(self):
        self.platforms = []
        y = self.HEIGHT - 30
        x = self.WIDTH / 2

        # Starting platform
        start_plat = pygame.Rect(x - 50, y, 100, self.PLATFORM_HEIGHT)
        self.platforms.append({
            'rect': start_plat, 'base_y': y, 'amplitude': 0, 'freq': 0, 'phase': 0
        })

        max_jump_height = self.JUMP_POWER_LARGE ** 2 / (2 * self.GRAVITY) * 0.8

        for i in range(1, self.NUM_PLATFORMS):
            y_gap = self.np_random.uniform(max_jump_height * 0.4, max_jump_height * 0.9)
            y -= y_gap

            x_offset = self.np_random.uniform(-150, 150)
            x = np.clip(x + x_offset, 50, self.WIDTH - 50)

            width = self.np_random.uniform(60, 120)

            amplitude = self.np_random.uniform(5, 20)
            freq = self.np_random.uniform(0.01, 0.05)
            phase = self.np_random.uniform(0, 2 * math.pi)

            plat = pygame.Rect(x - width / 2, y, width, self.PLATFORM_HEIGHT)
            self.platforms.append({
                'rect': plat, 'base_y': y, 'amplitude': amplitude, 'freq': freq, 'phase': phase
            })

        self.goal_platform = self.platforms[-1]

    def _update_platforms(self):
        for p in self.platforms:
            if p['amplitude'] > 0:
                p['rect'].y = p['base_y'] + math.sin(
                    self.steps * p['freq'] * self.platform_speed_multiplier + p['phase']) * p['amplitude']

    def _update_player(self, movement, shift_held):
        event_reward = 0

        # Handle jump action
        if self.on_platform and movement in [1, 3, 4]:  # 1=Up, 3=Left, 4=Right
            jump_power = self.JUMP_POWER_LARGE if shift_held else self.JUMP_POWER_SMALL
            self.player_vel[1] = -jump_power

            if movement == 3:  # Left
                self.player_vel[0] = -self.JUMP_HORIZONTAL_POWER
            elif movement == 4:  # Right
                self.player_vel[0] = self.JUMP_HORIZONTAL_POWER
            else:  # Up
                self.player_vel[0] = 0

            self.last_platform_y = self.on_platform['rect'].y
            self.on_platform = None
            self._create_particles(self.player_pos, 15, 'jump')
            # Sound: Player Jump

        # Apply physics
        if self.on_platform:
            # Stick to platform
            self.player_vel = [0, 0]
            self.player_pos[1] = self.on_platform['rect'].top - self.PLAYER_SIZE / 2
        else:
            # Gravity
            self.player_vel[1] += self.GRAVITY
            # Air friction
            self.player_vel[0] *= self.AIR_FRICTION

            self.player_pos[0] += self.player_vel[0]
            self.player_pos[1] += self.player_vel[1]

            # Screen bounds
            self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)

            # Check for landing
            if self.player_vel[1] > 0:
                player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE / 2,
                                          self.player_pos[1] - self.PLAYER_SIZE / 2, self.PLAYER_SIZE,
                                          self.PLAYER_SIZE)
                for p in self.platforms:
                    if player_rect.colliderect(p['rect']) and abs(player_rect.bottom - p['rect'].top) < self.player_vel[
                        1] + 1:
                        self.on_platform = p
                        self.player_pos[1] = p['rect'].top - self.PLAYER_SIZE / 2
                        self.player_vel[1] = 0
                        self._create_particles([self.player_pos[0], p['rect'].top], 10, 'land')
                        # Sound: Player Land

                        if p['rect'].y < self.last_platform_y:
                            event_reward += 1.0  # Reward for landing on a higher platform
                        break
        return event_reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0

        # Fell off screen
        if self.player_pos[1] > self.HEIGHT + self.PLAYER_SIZE:
            terminated = True
            terminal_reward = -5.0
            self.game_over = True

        # Reached goal
        if self.on_platform == self.goal_platform:
            terminated = True
            terminal_reward = 150.0
            self.game_over = True
            self.game_won = True

        # Time ran out
        if self.time_remaining <= 0 and not self.game_won:
            terminated = True
            self.game_over = True
            # No specific penalty, but the episode ends

        return terminated, terminal_reward

    def _create_particles(self, pos, count, p_type):
        for _ in range(count):
            if p_type == 'jump':
                vel = [self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(0.5, 2.5)]
            elif p_type == 'land':
                vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-1, 0)]
            else:  # generic
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)]

            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self._draw_background()

        # Render all game elements
        self._draw_particles()
        self._draw_platforms()
        self._draw_player()

        # Render UI overlay
        self._draw_ui()

        if self.render_mode == "human":
            if self.human_screen is not None:
                self.human_screen.blit(self.screen, (0, 0))
                pygame.display.flip()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _draw_platforms(self):
        for p in self.platforms:
            color = self.COLOR_GOAL if p == self.goal_platform else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, p['rect'], border_radius=3)
            # Add a slight 3D effect
            darker_color = tuple(c * 0.8 for c in color)
            pygame.draw.rect(self.screen, darker_color, p['rect'].move(0, 4), border_radius=3)

    def _draw_player(self):
        # Squash and stretch effect
        stretch_x = 1.0 - min(0.4, max(-0.4, self.player_vel[1] * 0.03))
        stretch_y = 1.0 + min(0.6, max(-0.4, self.player_vel[1] * 0.05))

        w = self.PLAYER_SIZE * stretch_x
        h = self.PLAYER_SIZE * stretch_y

        player_rect = pygame.Rect(
            int(self.player_pos[0] - w / 2),
            int(self.player_pos[1] - h / 2),
            int(w),
            int(h)
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # This part requires a surface with per-pixel alpha, which the main screen doesn't have.
        # It's better to skip it for compatibility or use a separate alpha surface.
        # For simplicity and to avoid potential issues, we'll draw a slightly larger rect instead.
        # pygame.gfxdraw.rectangle(self.screen, player_rect.inflate(4, 4), (*self.COLOR_PLAYER, 60))

    def _draw_particles(self):
        # Create a temporary surface for drawing particles with alpha
        particle_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*self.COLOR_PARTICLE, alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'] * (p['life'] / 20))
            if size > 0:
                pygame.draw.circle(particle_surface, color, pos, size)
        self.screen.blit(particle_surface, (0, 0))


    def _draw_ui(self):
        # Timer
        time_sec = max(0, self.time_remaining / self.FPS)
        timer_text = f"{time_sec:.1f}"
        text_surface = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 15, 10))

        # Score
        score_text = f"Score: {int(self.score)}"
        score_surface = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surface, (15, 10))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_GOAL if self.game_won else (255, 80, 80)
            end_text_surface = self.font_large.render(message, True, color)
            pos = (self.WIDTH / 2 - end_text_surface.get_width() / 2,
                   self.HEIGHT / 2 - end_text_surface.get_height() / 2)
            self.screen.blit(end_text_surface, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "player_y": self.player_pos[1]
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        # Human rendering is handled in _get_observation

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    done = False

    # --- Human player controls ---
    # This loop allows a human to play the game

    # Mapping from Pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Game loop
    total_reward = 0
    while not done:
        # Default action is no-op
        action = [0, 0, 0]  # movement=none, space=released, shift=released

        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Get pressed keys
        keys = pygame.key.get_pressed()

        # Movement
        move_key_pressed = False
        if keys[pygame.K_LEFT]:
            action[0] = 3
            move_key_pressed = True
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            move_key_pressed = True
        elif keys[pygame.K_UP]:
            action[0] = 1
            move_key_pressed = True

        # Shift for big jump
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render() # Updates the human display

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            done = True
            pygame.time.wait(2000) # Pause before closing

    env.close()