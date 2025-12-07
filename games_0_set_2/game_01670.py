
# Generated: 2025-08-27T17:53:09.177163
# Source Brief: brief_01670.md
# Brief Index: 1670

        
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
        "Controls: → to accelerate, ← to decelerate. ↑↓ to move vertically. Space for a speed boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a snail in a side-view race against an AI opponent. Avoid obstacles and reach the finish line first."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.TRACK_LENGTH = self.WIDTH * 8
        self.TRACK_TOP = 80
        self.TRACK_BOTTOM = self.HEIGHT - 50

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # Colors
        self.COLOR_BG = (135, 206, 235) # Sky Blue
        self.COLOR_TRACK = (34, 139, 34) # Forest Green
        self.COLOR_TRACK_LINE = (50, 205, 50) # Lime Green
        self.COLOR_OBSTACLE = (139, 69, 19) # Saddle Brown
        self.COLOR_OBSTACLE_OUTLINE = (101, 49, 0)
        self.COLOR_PLAYER = (255, 69, 0) # OrangeRed
        self.COLOR_PLAYER_SHELL = (255, 99, 71) # Tomato
        self.COLOR_OPPONENT = (65, 105, 225) # Royal Blue
        self.COLOR_OPPONENT_SHELL = (100, 149, 237) # Cornflower Blue
        self.COLOR_FINISH_LIGHT = (255, 255, 255)
        self.COLOR_FINISH_DARK = (0, 0, 0)
        self.COLOR_SPEED_LINE = (255, 255, 0, 150) # Yellow, semi-transparent
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # Game constants
        self.MAX_STEPS = 5000
        self.SNAIL_V_SPEED = 2.0
        self.SNAIL_ACCEL = 0.1
        self.SNAIL_DRAG = 0.985
        self.SNAIL_MAX_SPEED = 8.0
        self.BOOST_STRENGTH = 5.0
        self.BOOST_DURATION = 15 # steps
        self.COLLISION_PENALTY_DURATION = 30 # steps
        
        # Initialize state variables
        self.player_pos = None
        self.player_speed = None
        self.player_collisions = None
        self.boost_timer = None
        self.collision_timer = None
        self.player_trail = None

        self.opponent_pos = None
        self.opponent_speed = None
        self.opponent_base_speed = None
        self.opponent_target_y = None
        
        self.obstacles = None
        self.camera_x = None
        
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_status = None
        
        self.reset()
        
        # Run validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = [100.0, (self.TRACK_TOP + self.TRACK_BOTTOM) / 2]
        self.player_speed = 0.0
        self.player_collisions = 0
        self.boost_timer = 0
        self.collision_timer = 0
        self.player_trail = []
        
        # Opponent state
        self.opponent_pos = [100.0, (self.TRACK_TOP + self.TRACK_BOTTOM) / 2]
        self.opponent_speed = 1.0
        self.opponent_base_speed = 1.0
        self.opponent_target_y = self.opponent_pos[1]
        
        # World state
        self.camera_x = 0
        self._generate_obstacles()

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = "" # "win", "lose_race", "lose_crash"
        
        return self._get_observation(), self._get_info()

    def _generate_obstacles(self):
        self.obstacles = []
        for i in range(30):
            is_moving = self.np_random.random() > 0.7
            obstacle = {
                "pos": [
                    self.WIDTH + i * 250 + self.np_random.integers(0, 100),
                    self.np_random.integers(self.TRACK_TOP + 10, self.TRACK_BOTTOM - 10)
                ],
                "size": [self.np_random.integers(15, 25), self.np_random.integers(20, 40)],
                "moving": is_moving,
                "v_speed": (self.np_random.random() - 0.5) * 2 if is_moving else 0
            }
            self.obstacles.append(obstacle)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Not used

        reward = 0
        
        # === Player Logic ===
        prev_player_x = self.player_pos[0]
        
        # Vertical movement
        if movement == 1: self.player_pos[1] -= self.SNAIL_V_SPEED
        elif movement == 2: self.player_pos[1] += self.SNAIL_V_SPEED
        self.player_pos[1] = np.clip(self.player_pos[1], self.TRACK_TOP + 10, self.TRACK_BOTTOM - 10)

        # Horizontal movement
        if self.collision_timer == 0:
            if movement == 4: self.player_speed += self.SNAIL_ACCEL # Accelerate
            elif movement == 3: self.player_speed -= self.SNAIL_ACCEL # Decelerate
        
        # Apply boost
        if space_held and self.boost_timer == 0 and self.collision_timer == 0:
            self.player_speed += self.BOOST_STRENGTH
            self.boost_timer = self.BOOST_DURATION
            # sfx: boost sound

        # Apply drag and clamp speed
        self.player_speed *= self.SNAIL_DRAG
        self.player_speed = np.clip(self.player_speed, 0, self.SNAIL_MAX_SPEED)
        
        # Update position
        self.player_pos[0] += self.player_speed
        
        # Update timers
        if self.boost_timer > 0: self.boost_timer -= 1
        if self.collision_timer > 0: self.collision_timer -= 1

        # === Opponent AI Logic ===
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.opponent_base_speed = min(self.SNAIL_MAX_SPEED * 0.9, self.opponent_base_speed + 0.05)

        # Dodge obstacles
        scan_dist = 150
        for obs in self.obstacles:
            if self.opponent_pos[0] < obs["pos"][0] < self.opponent_pos[0] + scan_dist:
                obs_rect = pygame.Rect(obs["pos"][0], obs["pos"][1] - obs["size"][1]/2, obs["size"][0], obs["size"][1])
                if abs(obs_rect.centery - self.opponent_target_y) < obs_rect.height:
                    if self.opponent_target_y > self.HEIGHT / 2:
                        self.opponent_target_y = self.TRACK_TOP + 20
                    else:
                        self.opponent_target_y = self.TRACK_BOTTOM - 20
                    break
        
        # Move towards target Y
        if abs(self.opponent_pos[1] - self.opponent_target_y) > 1:
            self.opponent_pos[1] += np.sign(self.opponent_target_y - self.opponent_pos[1]) * self.SNAIL_V_SPEED * 0.5
        
        # Update speed and position
        self.opponent_speed = np.clip(self.opponent_base_speed + self.np_random.uniform(-0.5, 0.5), 0.5, self.SNAIL_MAX_SPEED)
        self.opponent_pos[0] += self.opponent_speed
        
        # === Obstacle Logic ===
        for obs in self.obstacles:
            if obs["moving"]:
                obs["pos"][1] += obs["v_speed"]
                if obs["pos"][1] < self.TRACK_TOP + obs["size"][1]/2 or obs["pos"][1] > self.TRACK_BOTTOM - obs["size"][1]/2:
                    obs["v_speed"] *= -1

        # === Collision Detection ===
        player_rect = pygame.Rect(self.player_pos[0] - 15, self.player_pos[1] - 10, 30, 20)
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs["pos"][0] - obs["size"][0]/2, obs["pos"][1] - obs["size"][1]/2, obs["size"][0], obs["size"][1])
            if player_rect.colliderect(obs_rect):
                self.player_collisions += 1
                self.player_speed = -1.0 # Bounce back
                self.collision_timer = self.COLLISION_PENALTY_DURATION
                reward -= 5.0
                obs["pos"][0] -= 50 # Move obstacle back to prevent multi-hits
                # sfx: crash sound
                break

        # === Rewards ===
        if self.player_pos[0] > prev_player_x:
            reward += 0.1 # Moving forward
        else:
            reward -= 0.2 # Staying still or moving backward
        
        if self.player_pos[0] < self.opponent_pos[0]:
            reward -= 0.5 # Behind opponent
        
        # === Termination Check ===
        terminated = False
        if self.player_collisions >= 3:
            terminated = True
            self.win_status = "lose_crash"
            # No terminal reward/penalty for crashing out
        elif self.player_pos[0] >= self.TRACK_LENGTH:
            terminated = True
            if self.opponent_pos[0] < self.TRACK_LENGTH:
                reward += 100.0 # Player wins
                self.win_status = "win"
            else:
                reward += 50.0 # Player finishes 2nd
                self.win_status = "lose_race"
        elif self.opponent_pos[0] >= self.TRACK_LENGTH:
            terminated = True
            reward -= 50.0 # Opponent wins, player loses
            self.win_status = "lose_race"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.win_status = "timeout"

        if terminated:
            self.game_over = True

        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        # Update camera to follow player
        self.camera_x = self.player_pos[0] - self.WIDTH / 4

        # Update player trail for speed lines
        if self.steps % 2 == 0:
            self.player_trail.append(list(self.player_pos))
            if len(self.player_trail) > 10:
                self.player_trail.pop(0)

        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw track
        track_rect = pygame.Rect(0, self.TRACK_TOP, self.WIDTH, self.TRACK_BOTTOM - self.TRACK_TOP)
        pygame.draw.rect(self.screen, self.COLOR_TRACK, track_rect)

        # Draw track lines for sense of speed
        for i in range(10):
            y = self.TRACK_TOP + i * (track_rect.height / 9)
            start_x = ((-self.camera_x * 1.5) % 100) - 100
            for j in range(int(self.WIDTH / 100) + 3):
                pygame.draw.line(self.screen, self.COLOR_TRACK_LINE, (start_x + j * 100, y), (start_x + j * 100 + 40, y), 1)

        # Draw finish line
        finish_x = self.TRACK_LENGTH - self.camera_x
        if finish_x < self.WIDTH + 50:
            pygame.draw.line(self.screen, self.COLOR_FINISH_DARK, (finish_x, self.TRACK_TOP), (finish_x, self.TRACK_BOTTOM), 5)
            for i in range(10):
                color = self.COLOR_FINISH_LIGHT if i % 2 == 0 else self.COLOR_FINISH_DARK
                pygame.draw.rect(self.screen, color, pygame.Rect(finish_x, self.TRACK_TOP + i * (track_rect.height/10), 15, track_rect.height/10))
                pygame.draw.rect(self.screen, color, pygame.Rect(finish_x+15, self.TRACK_TOP + (i+0.5) * (track_rect.height/10), 15, track_rect.height/10))

        # Draw obstacles
        for obs in self.obstacles:
            screen_x = obs["pos"][0] - self.camera_x
            if -50 < screen_x < self.WIDTH + 50:
                obs_rect = pygame.Rect(screen_x - obs["size"][0]/2, obs["pos"][1] - obs["size"][1]/2, obs["size"][0], obs["size"][1])
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect, border_radius=3)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, obs_rect, width=2, border_radius=3)

        # Draw speed lines
        if self.player_speed > 2.0 and len(self.player_trail) > 1:
            for i in range(len(self.player_trail) - 1):
                p1 = self.player_trail[i]
                p2 = self.player_trail[i+1]
                alpha = int(150 * (i / len(self.player_trail)))
                pygame.draw.line(self.screen, (*self.COLOR_SPEED_LINE[:3], alpha), 
                                 (p1[0] - self.camera_x - 15, p1[1]), 
                                 (p2[0] - self.camera_x - 15, p2[1]), 
                                 max(1, int(self.player_speed/3)))

        # Draw snails
        self._draw_snail(self.opponent_pos, self.COLOR_OPPONENT, self.COLOR_OPPONENT_SHELL)
        self._draw_snail(self.player_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_SHELL, is_player=True)

    def _draw_snail(self, pos, body_color, shell_color, is_player=False):
        screen_x = pos[0] - self.camera_x
        screen_y = pos[1]
        
        # Snail body
        pygame.gfxdraw.filled_ellipse(self.screen, int(screen_x), int(screen_y), 20, 10, body_color)
        pygame.gfxdraw.aaellipse(self.screen, int(screen_x), int(screen_y), 20, 10, body_color)
        
        # Snail shell
        shell_x, shell_y = int(screen_x - 10), int(screen_y - 10)
        pygame.gfxdraw.filled_circle(self.screen, shell_x, shell_y, 12, shell_color)
        pygame.gfxdraw.aacircle(self.screen, shell_x, shell_y, 12, shell_color)

        # Eye
        eye_x, eye_y = int(screen_x + 15), int(screen_y - 5)
        pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, eye_y), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (eye_x+1, eye_y), 2)
        
        # Effects for player
        if is_player:
            if self.collision_timer > 0 and self.steps % 4 < 2:
                # Flickering effect when hit
                pass # Don't draw to make it flicker
            if self.boost_timer > 0:
                # Glow effect when boosting
                glow_radius = 25 + 5 * math.sin(self.steps * 0.5)
                pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(screen_y), int(glow_radius), (*self.COLOR_PLAYER, 50))
                pygame.gfxdraw.aacircle(self.screen, int(screen_x), int(screen_y), int(glow_radius), (*self.COLOR_PLAYER, 50))

    def _render_ui(self):
        # UI Background for progress bar
        progress_bar_bg = pygame.Surface((self.WIDTH, 30), pygame.SRCALPHA)
        progress_bar_bg.fill((*self.COLOR_UI_BG[:3], 100))
        self.screen.blit(progress_bar_bg, (0, 0))
        
        # Progress bar
        player_progress = self.player_pos[0] / self.TRACK_LENGTH
        opponent_progress = self.opponent_pos[0] / self.TRACK_LENGTH
        
        player_x = int(player_progress * (self.WIDTH - 20)) + 10
        opp_x = int(opponent_progress * (self.WIDTH - 20)) + 10
        
        pygame.draw.rect(self.screen, self.COLOR_OPPONENT, (opp_x, 5, 10, 20), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (player_x, 5, 10, 20), border_radius=3)
        
        # Collision counter
        hits_text = self.font_small.render(f"Hits: {self.player_collisions} / 3", True, self.COLOR_UI_TEXT)
        self.screen.blit(hits_text, (10, self.HEIGHT - 30))

        # Speed display
        speed_text = self.font_small.render(f"Speed: {self.player_speed:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (self.WIDTH - 120, self.HEIGHT - 30))
        
        # Game over text
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            if self.win_status == "win":
                msg = "YOU WIN!"
                color = (0, 255, 0)
            elif self.win_status == "lose_race":
                msg = "YOU LOST THE RACE!"
                color = (255, 100, 0)
            elif self.win_status == "lose_crash":
                msg = "CRASHED OUT!"
                color = (255, 0, 0)
            else: # timeout
                msg = "TIME'S UP!"
                color = (255, 255, 0)
                
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player_pos[0],
            "opponent_x": self.opponent_pos[0],
            "collisions": self.player_collisions,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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
    
    # To hold down keys
    key_map = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
    }

    # Use a separate screen for rendering if playing directly
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Snail Race")
    
    # Game loop
    running = True
    while running:
        # --- Action mapping from keyboard ---
        movement = 0 # none
        if key_map[pygame.K_UP]: movement = 1
        elif key_map[pygame.K_DOWN]: movement = 2
        elif key_map[pygame.K_LEFT]: movement = 3
        elif key_map[pygame.K_RIGHT]: movement = 4
        
        space = 1 if key_map[pygame.K_SPACE] else 0
        shift = 1 if key_map[pygame.K_LSHIFT] else 0
        
        action = [movement, space, shift]

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_map:
                    key_map[event.key] = True
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    done = False
            if event.type == pygame.KEYUP:
                if event.key in key_map:
                    key_map[event.key] = False

        # --- Gym step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # Convert the observation (which is what the agent sees) to a format pygame can display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # 30 FPS

    env.close()