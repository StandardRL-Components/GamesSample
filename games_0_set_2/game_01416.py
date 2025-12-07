
# Generated: 2025-08-27T17:03:54.575681
# Source Brief: brief_01416.md
# Brief Index: 1416

        
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
        "Controls: ↑↓ to aim, ←→ to set power. Press space to shoot. Press shift to reset aim."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist side-view basketball free throw game. Sink five consecutive baskets to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (44, 62, 80)  # #2c3e50
    COLOR_FLOOR = (89, 69, 69)  # #594545
    COLOR_BALL = (230, 126, 34)  # #e67e22
    COLOR_HOOP = (192, 57, 43)  # #c0392b
    COLOR_BACKBOARD = (149, 165, 166)  # #95a5a6
    COLOR_NET = (189, 195, 199)  # #bdc3c7
    COLOR_TEXT = (236, 240, 241)  # #ecf0f1
    COLOR_POWER_BAR_FILL = (46, 204, 113)  # #2ecc71
    COLOR_POWER_BAR_BG = (52, 73, 94)  # #34495e

    # Screen Dimensions
    WIDTH, HEIGHT = 640, 400

    # Game Parameters
    WIN_STREAK = 5
    MAX_STEPS = 1000
    FPS = 30
    SHOT_CLOCK_SECONDS = 3
    SHOT_CLOCK_FRAMES = SHOT_CLOCK_SECONDS * FPS
    
    # Physics
    GRAVITY = 0.5
    BALL_RADIUS = 12
    BALL_START_POS = (100, 300)
    BOUNCE_DAMPING = 0.7
    
    # Hoop Geometry
    HOOP_POS = (540, 150)
    HOOP_WIDTH = 50
    HOOP_THICKNESS = 4
    BACKBOARD_RECT = pygame.Rect(HOOP_POS[0] + 10, HOOP_POS[1] - 50, 10, 100)
    RIM_FRONT_POS = (HOOP_POS[0] - HOOP_WIDTH // 2, HOOP_POS[1])
    RIM_BACK_POS = (HOOP_POS[0] + HOOP_WIDTH // 2, HOOP_POS[1])
    RIM_RADIUS = 5

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 40, bold=True)
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        # self.validate_implementation() # Comment out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.consecutive_baskets = 0
        self.last_reward = 0
        
        self._reset_shot()
        self.message = ""
        self.message_timer = 0
        
        return self._get_observation(), self._get_info()

    def _reset_shot(self):
        """Resets the state for a new shot attempt."""
        self.ball_state = "AIMING"  # AIMING, FLYING, RESOLVING
        self.ball_pos = list(self.BALL_START_POS)
        self.ball_vel = [0, 0]
        self.ball_trail = []
        
        self.aim_angle = 60  # degrees from horizontal
        self.aim_power = 50  # 0-100
        
        self.shot_clock = self.SHOT_CLOCK_FRAMES
        self.resolve_timer = 0
        self.scored_this_frame = False

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean, use as a trigger
        shift_held = action[2] == 1  # Boolean
        
        self.last_reward = 0
        self.scored_this_frame = False

        if not self.game_over:
            if self.ball_state == "AIMING":
                self._handle_aiming(movement, space_pressed, shift_held)
            elif self.ball_state == "FLYING":
                self._handle_flying()
            elif self.ball_state == "RESOLVING":
                self._handle_resolving()
        
        self.steps += 1
        
        # Update UI messages
        if self.message_timer > 0:
            self.message_timer -= 1
        else:
            self.message = ""

        reward = self.last_reward
        self.score += reward
        terminated = self._check_termination()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_aiming(self, movement, space_pressed, shift_held):
        self.shot_clock -= 1
        
        if shift_held:
            self.aim_angle = 60
            self.aim_power = 50
        else:
            if movement == 1: self.aim_angle += 1   # Up
            if movement == 2: self.aim_angle -= 1   # Down
            if movement == 3: self.aim_power -= 1   # Left
            if movement == 4: self.aim_power += 1   # Right
        
        self.aim_angle = np.clip(self.aim_angle, 20, 90)
        self.aim_power = np.clip(self.aim_power, 0, 100)

        if space_pressed or self.shot_clock <= 0:
            self._take_shot()

    def _take_shot(self):
        self.ball_state = "FLYING"
        speed = 2 + (self.aim_power / 100) * 20
        angle_rad = math.radians(self.aim_angle)
        self.ball_vel = [
            speed * math.cos(angle_rad),
            -speed * math.sin(angle_rad)
        ]
        # Sound: Shoot ball

    def _handle_flying(self):
        # Update trail
        if self.steps % 2 == 0:
            self.ball_trail.append(tuple(self.ball_pos))
            if len(self.ball_trail) > 15:
                self.ball_trail.pop(0)

        # Physics
        self.ball_vel[1] += self.GRAVITY
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # Check for score
        if (self.ball_pos[1] > self.HOOP_POS[1] - self.BALL_RADIUS and
            self.ball_pos[1] - self.ball_vel[1] <= self.HOOP_POS[1] - self.BALL_RADIUS and
            self.ball_pos[0] > self.RIM_FRONT_POS[0] and
            self.ball_pos[0] < self.RIM_BACK_POS[0] and
            self.ball_vel[1] > 0):
            self._handle_score()
            return

        # Check collisions
        self._check_collisions()

        # Check for miss (out of bounds)
        if self.ball_pos[1] > self.HEIGHT - 30 or self.ball_pos[0] < 0 or self.ball_pos[0] > self.WIDTH:
            self._handle_miss()
            # Sound: Ball bounce / miss

    def _check_collisions(self):
        # Backboard
        if self.ball_pos[0] + self.BALL_RADIUS > self.BACKBOARD_RECT.left and self.ball_vel[0] > 0:
            if self.BACKBOARD_RECT.colliderect(
                self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS,
                self.BALL_RADIUS * 2, self.BALL_RADIUS * 2):
                self.ball_pos[0] = self.BACKBOARD_RECT.left - self.BALL_RADIUS
                self.ball_vel[0] *= -self.BOUNCE_DAMPING
                self.last_reward -= 0.1
                # Sound: Backboard hit

        # Rims
        for rim_pos in [self.RIM_FRONT_POS, self.RIM_BACK_POS]:
            dist = math.hypot(self.ball_pos[0] - rim_pos[0], self.ball_pos[1] - rim_pos[1])
            if dist < self.BALL_RADIUS + self.RIM_RADIUS:
                # Simple vertical bounce for game feel
                self.ball_vel[1] *= -self.BOUNCE_DAMPING
                self.ball_vel[0] *= self.BOUNCE_DAMPING
                # Move ball slightly away to prevent sticking
                self.ball_pos[1] += -1 if self.ball_vel[1] > 0 else 1
                self.last_reward -= 0.01
                # Sound: Rim hit
                break

    def _handle_score(self):
        self.scored_this_frame = True
        self.consecutive_baskets += 1
        self.last_reward += 1.0
        self.message = "SWISH!"
        self.message_timer = self.FPS
        self.ball_state = "RESOLVING"
        self.resolve_timer = self.FPS // 2 # 0.5s pause
        # Sound: Swish

        if self.consecutive_baskets >= self.WIN_STREAK:
            self.last_reward += 100.0
            self.message = "YOU WIN!"
            self.message_timer = self.FPS * 3
            self.game_over = True
        else:
            self.last_reward += 0.1

    def _handle_miss(self):
        self.last_reward -= 10.0
        self.message = f"STREAK BROKEN: {self.consecutive_baskets}"
        self.message_timer = self.FPS * 2
        self.consecutive_baskets = 0
        self.ball_state = "RESOLVING"
        self.resolve_timer = self.FPS # 1s pause
        self.game_over = True

    def _handle_resolving(self):
        self.resolve_timer -= 1
        if self.resolve_timer <= 0 and not self.game_over:
            self._reset_shot()

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Floor
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, (0, self.HEIGHT - 30, self.WIDTH, 30))
        
        # Backboard
        pygame.draw.rect(self.screen, self.COLOR_BACKBOARD, self.BACKBOARD_RECT)

        # Net (visual only)
        if self.scored_this_frame:
            net_depth = 20 # Animate stretch
        else:
            net_depth = 0
        for i in range(6):
            x_start = self.RIM_FRONT_POS[0] + i * (self.HOOP_WIDTH / 5)
            pygame.draw.line(self.screen, self.COLOR_NET, (x_start, self.RIM_FRONT_POS[1]), 
                             (self.HOOP_POS[0], self.HOOP_POS[1] + 30 + net_depth), 1)

        # Hoop Rims
        pygame.draw.ellipse(self.screen, self.COLOR_HOOP, (self.RIM_FRONT_POS[0] - self.HOOP_THICKNESS, self.RIM_FRONT_POS[1]-self.HOOP_THICKNESS, self.HOOP_WIDTH + 2*self.HOOP_THICKNESS, 2*self.HOOP_THICKNESS))
        pygame.draw.ellipse(self.screen, self.COLOR_BG, (self.RIM_FRONT_POS[0], self.RIM_FRONT_POS[1]-self.HOOP_THICKNESS+1, self.HOOP_WIDTH, 2*self.HOOP_THICKNESS-2))

        # Ball Trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(255 * (i / len(self.ball_trail)))
            trail_color = self.COLOR_BALL + (alpha,)
            s = pygame.Surface((self.BALL_RADIUS*2, self.BALL_RADIUS*2), pygame.SRCALPHA)
            pygame.draw.circle(s, trail_color, (self.BALL_RADIUS, self.BALL_RADIUS), self.BALL_RADIUS * (i / len(self.ball_trail)))
            self.screen.blit(s, (int(pos[0] - self.BALL_RADIUS), int(pos[1] - self.BALL_RADIUS)))

        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)

        # Aiming guide
        if self.ball_state == "AIMING":
            self._render_aim_guide()

    def _render_aim_guide(self):
        speed = 2 + (self.aim_power / 100) * 20
        angle_rad = math.radians(self.aim_angle)
        vel = [speed * math.cos(angle_rad), -speed * math.sin(angle_rad)]
        pos = list(self.ball_pos)
        
        points = []
        for _ in range(20): # Simulate 20 frames of trajectory
            vel[1] += self.GRAVITY
            pos[0] += vel[0]
            pos[1] += vel[1]
            if _ % 2 == 0:
                points.append(tuple(map(int, pos)))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_TEXT, False, points, 1)

    def _render_ui(self):
        # Score and Streak
        score_surf = self.font_ui.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        streak_surf = self.font_ui.render(f"STREAK: {self.consecutive_baskets} / {self.WIN_STREAK}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(streak_surf, (10, 35))
        
        # Shot Clock
        time_left = max(0, self.shot_clock / self.FPS)
        clock_color = self.COLOR_HOOP if time_left < 1.0 else self.COLOR_TEXT
        clock_surf = self.font_ui.render(f"TIME: {time_left:.1f}", True, clock_color)
        self.screen.blit(clock_surf, (self.WIDTH - clock_surf.get_width() - 10, 10))

        # Power and Angle Bars
        if self.ball_state == "AIMING":
            # Power Bar
            bar_w, bar_h = 150, 20
            bar_x, bar_y = self.BALL_START_POS[0] - bar_w/2, self.BALL_START_POS[1] + 40
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
            fill_w = bar_w * (self.aim_power / 100)
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FILL, (bar_x, bar_y, fill_w, bar_h))
            
            # Angle Text
            angle_surf = self.font_ui.render(f"{self.aim_angle:.0f}°", True, self.COLOR_TEXT)
            self.screen.blit(angle_surf, (bar_x + bar_w + 10, bar_y))

        # Central Message
        if self.message_timer > 0:
            msg_surf = self.font_msg.render(self.message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "consecutive_baskets": self.consecutive_baskets,
        }

    def close(self):
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Controls ---
    # Map keyboard keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame window for human play
    render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Construct the action based on keyboard state
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Only one movement at a time
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_SHIFT]:
            shift_action = 1
        
        action = [movement_action, space_action, shift_action]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle game over and quit events
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Streak: {info['consecutive_baskets']}")
            obs, info = env.reset()
            # Add a small delay before restarting
            pygame.time.wait(2000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)

    env.close()