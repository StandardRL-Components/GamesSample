
# Generated: 2025-08-28T03:41:35.733680
# Source Brief: brief_05006.md
# Brief Index: 5006

        
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
    """
    An arcade puzzler where the player controls a pixel, navigating a series of
    vertically moving platforms to reach a goal. The game features physics-based
    movement, a time limit, and progressively harder levels.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Avoid falling off the screen and reach the green goal before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push a pixel across a shifting landscape to reach the goal. A fast-paced arcade puzzler that balances speed and precision."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and rendering constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PLAYER = (255, 64, 64)
        self.COLOR_GOAL = (64, 255, 64)
        self.COLOR_PLATFORMS = [(0, 150, 255), (255, 255, 0), (0, 255, 255)]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE = (255, 150, 150)

        # Physics and game constants
        self.GRAVITY = 0.4
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = -0.15
        self.JUMP_STRENGTH = -9
        self.MAX_VEL_X = 5
        self.PLAYER_SIZE = 12
        self.LEVEL_TIME_SECONDS = 60
        self.TOTAL_LEVELS = 3
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.is_grounded = None
        self.platforms = None
        self.goal_rect = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.level = None
        self.time_remaining = None
        self.last_dist_to_goal = None
        self.level_complete_timer = None
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.game_over = False
        self.level = 1
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes the state for the current level."""
        self.steps = 0
        self.time_remaining = self.LEVEL_TIME_SECONDS * self.FPS
        self.particles = []
        self.level_complete_timer = 0
        
        level_configs = {
            1: {
                "player_start": (50, 100),
                "goal": pygame.Rect(580, 350, 40, 40),
                "platforms": [
                    # x, y_base, width, amp, freq_mult, color_idx
                    (0, 380, 120, 0, 0, 0),
                    (150, 300, 100, 50, 1.0, 1),
                    (300, 250, 100, 70, 1.2, 2),
                    (450, 320, 100, 60, 0.8, 0),
                ]
            },
            2: {
                "player_start": (50, 300),
                "goal": pygame.Rect(580, 80, 40, 40),
                "platforms": [
                    (0, 380, 120, 0, 0, 0),
                    (150, 320, 80, 80, 1.1, 1),
                    (250, 220, 80, 100, 1.3, 2),
                    (380, 150, 80, 40, 1.0, 0),
                    (500, 250, 60, 120, 1.5, 1),
                ]
            },
            3: {
                "player_start": (300, 50),
                "goal": pygame.Rect(20, 350, 40, 40),
                "platforms": [
                    (260, 100, 80, 0, 0, 0),
                    (450, 150, 80, 60, 1.8, 1),
                    (100, 180, 80, 70, 2.0, 2),
                    (280, 280, 80, 90, 1.6, 0),
                    (480, 350, 80, 40, 2.2, 1),
                    (100, 380, 120, 0, 0, 2),
                ]
            }
        }
        
        config = level_configs[self.level]
        self.player_pos = list(config["player_start"])
        self.player_vel = [0, 0]
        self.is_grounded = False
        self.goal_rect = config["goal"]
        
        self.platforms = []
        platform_speed_multiplier = 0.5 + (self.level - 1) * 0.1
        for p in config["platforms"]:
            self.platforms.append({
                "rect": pygame.Rect(p[0], p[1], p[2], 15),
                "base_y": p[1],
                "amp": p[3],
                "freq": 0.02 * p[4] * platform_speed_multiplier,
                "phase": self.np_random.random() * 2 * math.pi,
                "color": self.COLOR_PLATFORMS[p[5]]
            })

        self.last_dist_to_goal = self._get_dist_to_goal()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update timers
        self.steps += 1
        self.time_remaining -= 1
        if self.level_complete_timer > 0:
            self.level_complete_timer -= 1
            if self.level_complete_timer == 0:
                self._handle_level_completion()
            # While showing "Level Complete", freeze gameplay but allow rendering
            return self._get_observation(), 0, self.game_over, False, self._get_info()

        # Process action
        movement, _, _ = action # space and shift are unused
        
        # Horizontal movement
        accel = [0, self.GRAVITY]
        if movement == 3:  # Left
            accel[0] -= self.PLAYER_ACCEL
        if movement == 4:  # Right
            accel[0] += self.PLAYER_ACCEL
        
        # Jumping
        if movement == 1 and self.is_grounded: # Up
            self.player_vel[1] = self.JUMP_STRENGTH
            self.is_grounded = False
            # sfx: jump

        # Apply friction and acceleration
        accel[0] += self.player_vel[0] * self.PLAYER_FRICTION
        self.player_vel[0] += accel[0]
        self.player_vel[1] += accel[1]
        
        # Clamp horizontal velocity
        self.player_vel[0] = max(-self.MAX_VEL_X, min(self.MAX_VEL_X, self.player_vel[0]))
        
        # Update player position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Update platforms
        for p in self.platforms:
            p["rect"].y = p["base_y"] + p["amp"] * math.sin(p["freq"] * self.steps + p["phase"])

        # Collision detection
        self.is_grounded = False
        # Player vs Platforms
        for p in self.platforms:
            if player_rect.colliderect(p["rect"]):
                # Check if landing on top
                if self.player_vel[1] > 0 and player_rect.bottom - self.player_vel[1] <= p["rect"].top + 1:
                    self.player_pos[1] = p["rect"].top - self.PLAYER_SIZE
                    self.player_vel[1] = 0
                    self.is_grounded = True
                    # sfx: land
                    break
        
        # Player vs Screen bounds
        if self.player_pos[0] < 0:
            self.player_pos[0] = 0
            self.player_vel[0] = 0
        if self.player_pos[0] > self.WIDTH - self.PLAYER_SIZE:
            self.player_pos[0] = self.WIDTH - self.PLAYER_SIZE
            self.player_vel[0] = 0
        
        # Update particles
        if abs(self.player_vel[0]) > 0.1 and self.is_grounded:
            self._spawn_particle()
        self._update_particles()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.score += reward
        
        # Check termination conditions
        terminated = self._check_termination(player_rect)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_dist_to_goal(self):
        player_center = (self.player_pos[0] + self.PLAYER_SIZE / 2, self.player_pos[1] + self.PLAYER_SIZE / 2)
        goal_center = self.goal_rect.center
        return math.hypot(player_center[0] - goal_center[0], player_center[1] - goal_center[1])

    def _calculate_reward(self):
        # Continuous reward for getting closer to the goal
        current_dist = self._get_dist_to_goal()
        reward = (self.last_dist_to_goal - current_dist) * 0.1
        self.last_dist_to_goal = current_dist
        return reward

    def _check_termination(self, player_rect):
        terminated = False
        reward_bonus = 0

        # Fell off screen
        if self.player_pos[1] > self.HEIGHT:
            terminated = True
            self.game_over = True
            # sfx: fail

        # Time ran out
        if self.time_remaining <= 0:
            terminated = True
            self.game_over = True
            # sfx: timeout

        # Reached goal
        if player_rect.colliderect(self.goal_rect):
            reward_bonus += 5.0 # Small bonus for touching
            if self.level == self.TOTAL_LEVELS:
                reward_bonus += 300.0 # Win game bonus
                self.game_over = True
                terminated = True
            else:
                reward_bonus += 100.0 # Level complete bonus
            
            self.score += reward_bonus
            self.level_complete_timer = self.FPS * 1.5 # Show message for 1.5s
            # sfx: goal_reached

        return terminated

    def _handle_level_completion(self):
        """Advances to the next level."""
        self.level += 1
        self._setup_level()
    
    def _spawn_particle(self):
        if len(self.particles) < 50:
            p_pos = [self.player_pos[0] + self.PLAYER_SIZE/2, self.player_pos[1] + self.PLAYER_SIZE]
            p_vel = [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-1, -0.5)]
            p_life = self.np_random.integers(10, 20)
            self.particles.append({"pos": p_pos, "vel": p_vel, "life": p_life, "max_life": p_life})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render platforms
        for p in self.platforms:
            pygame.draw.rect(self.screen, p["color"], p["rect"], border_radius=3)
        
        # Render goal (flashing)
        flash_alpha = 128 + 127 * math.sin(self.steps * 0.2)
        goal_surface = pygame.Surface(self.goal_rect.size, pygame.SRCALPHA)
        goal_surface.fill((*self.COLOR_GOAL, flash_alpha))
        self.screen.blit(goal_surface, self.goal_rect.topleft)
        pygame.gfxdraw.rectangle(self.screen, self.goal_rect, self.COLOR_GOAL)
        
        # Render particles
        for p in self.particles:
            size = max(0, (p["life"] / p["max_life"]) * 4)
            color = self.COLOR_PARTICLE
            pygame.draw.rect(self.screen, color, (int(p["pos"][0]), int(p["pos"][1]), int(size), int(size)))

        # Render player
        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        # Level text
        level_text = self.font_ui.render(f"Level: {self.level}/{self.TOTAL_LEVELS}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Timer text
        time_str = f"Time: {max(0, self.time_remaining // self.FPS):02d}"
        timer_text = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Level complete / Game over messages
        if self.level_complete_timer > 0:
            if self.game_over:
                msg = "YOU WIN!"
            else:
                msg = f"LEVEL {self.level} COMPLETE!"
            
            msg_text = self.font_msg.render(msg, True, self.COLOR_TEXT)
            pos = (self.WIDTH // 2 - msg_text.get_width() // 2, self.HEIGHT // 2 - msg_text.get_height() // 2)
            self.screen.blit(msg_text, pos)
        elif self.game_over:
            msg = "GAME OVER"
            msg_text = self.font_msg.render(msg, True, self.COLOR_PLAYER)
            pos = (self.WIDTH // 2 - msg_text.get_width() // 2, self.HEIGHT // 2 - msg_text.get_height() // 2)
            self.screen.blit(msg_text, pos)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_remaining_steps": self.time_remaining,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Pixel Pusher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # Action mapping for human input
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # space and shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            print("Press 'R' to restart or close the window to quit.")
            
        clock.tick(env.FPS)
        
    env.close()