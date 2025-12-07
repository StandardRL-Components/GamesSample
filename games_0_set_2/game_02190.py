
# Generated: 2025-08-27T19:34:18.669361
# Source Brief: brief_02190.md
# Brief Index: 2190

        
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
        "Use ← and → to jump left and right. Use ↑ for a high vertical jump and ↓ for a short hop. "
        "Hold Space for extra height and Shift for extra distance."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Leap between floating platforms to reach the goal at the top. Master the art of the jump "
        "to get the highest score, but don't fall!"
    )

    # Frames only advance when an action is received.
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 10)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    COLOR_PLATFORM = (0, 200, 100)
    COLOR_PLATFORM_GOAL = (255, 100, 100)
    COLOR_PLATFORM_OUTLINE = (0, 100, 50)
    COLOR_UI_TEXT = (255, 255, 255)
    PARTICLE_COLORS = [(255, 200, 0), (255, 150, 0), (255, 255, 100)]

    # Physics
    GRAVITY = 0.35
    JUMP_VEL_V = -8.0
    JUMP_VEL_H = 5.0
    SPACE_BOOST = -3.0
    SHIFT_BOOST = 3.0
    PLAYER_RADIUS = 10
    PLATFORM_HEIGHT = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.prev_player_pos = np.array([0.0, 0.0])
        self.on_platform = False
        self.platforms = []
        self.particles = []
        self.highest_platform_idx = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_platforms()

        start_platform = self.platforms[0]
        self.player_pos = np.array([float(start_platform.centerx), float(start_platform.top - self.PLAYER_RADIUS)])
        self.player_vel = np.array([0.0, 0.0])
        self.prev_player_pos = self.player_pos.copy()
        self.on_platform = True
        self.highest_platform_idx = 0
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        # --- Handle Input (Jumping) ---
        if self.on_platform and movement != 0:
            self.on_platform = False
            jump_vel = np.array([0.0, 0.0])

            # Base jump velocity from movement direction
            if movement == 1: # Up
                jump_vel[1] = self.JUMP_VEL_V * 1.2 # Stronger vertical
            elif movement == 2: # Down (short hop)
                jump_vel[1] = self.JUMP_VEL_V * 0.6
            elif movement == 3: # Left
                jump_vel = np.array([-self.JUMP_VEL_H, self.JUMP_VEL_V])
            elif movement == 4: # Right
                jump_vel = np.array([self.JUMP_VEL_H, self.JUMP_VEL_V])

            # Apply modifiers
            if space_held:
                jump_vel[1] += self.SPACE_BOOST
            if shift_held:
                if jump_vel[0] > 0: jump_vel[0] += self.SHIFT_BOOST
                elif jump_vel[0] < 0: jump_vel[0] -= self.SHIFT_BOOST

            self.player_vel = jump_vel
            # sfx: jump.wav

        # --- Update Physics ---
        self.prev_player_pos = self.player_pos.copy()
        if not self.on_platform:
            self.player_vel[1] += self.GRAVITY
            self.player_pos += self.player_vel

        # --- Screen Boundaries ---
        if self.player_pos[0] < self.PLAYER_RADIUS:
            self.player_pos[0] = self.PLAYER_RADIUS
            self.player_vel[0] = 0
        if self.player_pos[0] > self.SCREEN_WIDTH - self.PLAYER_RADIUS:
            self.player_pos[0] = self.SCREEN_WIDTH - self.PLAYER_RADIUS
            self.player_vel[0] = 0

        # --- Collision Detection ---
        landed_this_frame = False
        if not self.on_platform and self.player_vel[1] > 0:
            player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_RADIUS, self.player_pos[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
            for i, plat in enumerate(self.platforms):
                prev_bottom = self.prev_player_pos[1] + self.PLAYER_RADIUS
                curr_bottom = self.player_pos[1] + self.PLAYER_RADIUS

                # Check for landing: was above, now intersecting, and horizontally aligned
                if (prev_bottom <= plat.top and curr_bottom >= plat.top and
                    player_rect.left < plat.right and player_rect.right > plat.left):

                    self.player_pos[1] = plat.top - self.PLAYER_RADIUS
                    self.player_vel = np.array([0.0, 0.0])
                    self.on_platform = True
                    landed_this_frame = True
                    self._create_particles(self.player_pos[0], plat.top)
                    # sfx: land.wav

                    # Reward for reaching a new, higher platform
                    if i > self.highest_platform_idx:
                        reward += 1.0 * (i - self.highest_platform_idx)
                        self.score += (i - self.highest_platform_idx)
                        self.highest_platform_idx = i
                    
                    # Check for reaching the goal
                    if i == len(self.platforms) - 1:
                        reward += 100.0
                        self.score += 100
                        terminated = True
                        # sfx: win.wav
                    
                    break

        # --- Reward for staying on a platform ---
        if self.on_platform and not landed_this_frame:
             reward += 0.01

        # --- Update Particles ---
        self._update_particles()

        # --- Check Termination Conditions ---
        self.steps += 1
        if self.player_pos[1] - self.PLAYER_RADIUS > self.SCREEN_HEIGHT:
            reward = -100.0
            terminated = True
            # sfx: fall.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_platforms(self):
        self.platforms = []
        # Starting platform
        start_plat = pygame.Rect(self.SCREEN_WIDTH / 2 - 50, self.SCREEN_HEIGHT - 50, 100, self.PLATFORM_HEIGHT)
        self.platforms.append(start_plat)

        # Procedurally generate intermediate platforms
        num_platforms = 12
        current_y = start_plat.y
        for i in range(num_platforms):
            last_plat = self.platforms[-1]
            max_h_dist = 120 + i * 5 # progressively harder
            max_v_dist = 100
            min_v_dist = 40

            px = last_plat.centerx
            py = last_plat.y
            
            w = self.np_random.integers(60, 100)
            x = px + self.np_random.uniform(-max_h_dist, max_h_dist)
            y = py - self.np_random.uniform(min_v_dist, max_v_dist)
            
            # Ensure platforms are reachable and within bounds
            x = np.clip(x, w/2, self.SCREEN_WIDTH - w/2)
            y = max(y, 60) # Don't go too close to the top

            new_plat = pygame.Rect(x - w / 2, y, w, self.PLATFORM_HEIGHT)
            self.platforms.append(new_plat)

        # Goal platform
        last_gen_plat = self.platforms[-1]
        goal_y = max(20, last_gen_plat.y - 80)
        goal_plat = pygame.Rect(self.SCREEN_WIDTH/2 - 60, goal_y, 120, self.PLATFORM_HEIGHT + 5)
        self.platforms.append(goal_plat)

    def _create_particles(self, x, y):
        for _ in range(20):
            vel = np.array([self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 0)])
            lifespan = self.np_random.integers(15, 30)
            color = random.choice(self.PARTICLE_COLORS)
            self.particles.append({'pos': np.array([x, y]), 'vel': vel, 'life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['vel'][1] += self.GRAVITY * 0.2
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.SCREEN_HEIGHT):
            mix = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - mix) + self.COLOR_BG_BOTTOM[0] * mix,
                self.COLOR_BG_TOP[1] * (1 - mix) + self.COLOR_BG_BOTTOM[1] * mix,
                self.COLOR_BG_TOP[2] * (1 - mix) + self.COLOR_BG_BOTTOM[2] * mix,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw platforms
        for i, plat in enumerate(self.platforms):
            color = self.COLOR_PLATFORM_GOAL if i == len(self.platforms) - 1 else self.COLOR_PLATFORM
            pygame.draw.rect(self.screen, color, plat, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, plat, 2, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            color_with_alpha = p['color'] + (alpha,)
            temp_surf = pygame.Surface((2, 2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color_with_alpha, temp_surf.get_rect())
            self.screen.blit(temp_surf, (int(p['pos'][0]), int(p['pos'][1])))

        # Draw player glow
        glow_radius = int(self.PLAYER_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(self.player_pos[0] - glow_radius), int(self.player_pos[1] - glow_radius)))
        
        # Draw player
        player_pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "GOAL REACHED!" if self.highest_platform_idx == len(self.platforms) - 1 else "GAME OVER"
            status_render = self.font.render(status_text, True, self.COLOR_UI_TEXT)
            status_rect = status_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(status_render, status_rect)

            final_score_render = self.small_font.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            final_score_rect = final_score_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_render, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "player_vel": self.player_vel.tolist(),
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Platformer Game")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("--- Human Controls ---")
    print(GameEnv.user_guide)
    
    while not done:
        # --- Action mapping for human keyboard input ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_r: # Reset on 'R' key
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        # --- Step the environment ---
        # Only step if an action is taken or if the player is in the air
        if movement != 0 or not env.on_platform:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Episode finished. Total reward: {total_reward}")
                # Wait for a moment before auto-resetting or quitting
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS for human play

    env.close()