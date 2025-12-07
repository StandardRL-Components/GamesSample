
# Generated: 2025-08-27T13:04:19.151009
# Source Brief: brief_00250.md
# Brief Index: 250

        
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
    A minimalist side-scrolling arcade game where the player uses a single button
    to jump and navigate a procedurally generated obstacle course.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Hold SPACE to jump higher. Avoid the red obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist side-scrolling arcade game. Jump to navigate a "
        "procedurally generated obstacle course against the clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.TIME_LIMIT_SECONDS = 10
        self.MAX_EPISODE_STEPS = self.FPS * self.TIME_LIMIT_SECONDS + 10 # A bit of buffer

        # Player physics
        self.GRAVITY = 0.4
        self.JUMP_STRENGTH = -9.0
        self.JUMP_SUSTAIN_FORCE = 0.15
        self.MAX_JUMP_HOLD_FRAMES = 18
        self.PLAYER_HORIZONTAL_POS = 100
        self.PLAYER_SIZE = 20
        self.WORLD_SPEED = 6

        # Level generation
        self.LEVEL_LENGTH = 6000 # Distance to finish line
        self.OBSTACLE_MIN_WIDTH = 30
        self.OBSTACLE_MAX_WIDTH = 80
        self.OBSTACLE_MIN_HEIGHT = 40
        self.OBSTACLE_MAX_HEIGHT = 150
        self.INITIAL_GAP_SIZE = self.PLAYER_SIZE * 8
        self.FINAL_GAP_SIZE = self.PLAYER_SIZE * 3.5

        # Colors
        self.COLOR_BG_TOP = (20, 30, 80)
        self.COLOR_BG_BOTTOM = (60, 80, 150)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_OBSTACLE_OUTLINE = (180, 50, 50)
        self.COLOR_FINISH_LINE = (80, 255, 80)
        self.COLOR_FLOOR = (180, 190, 220)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        self.font_tiny = pygame.font.Font(None, 24)

        # Internal state variables
        self.player_pos = None
        self.player_vel_y = None
        self.on_ground = None
        self.jump_hold_frames = None
        self.obstacles = None
        self.finish_line_x = None
        self.world_scroll_x = None
        self.particles = None
        self.steps = None
        self.score = None
        self.time_left = None
        self.game_over = None
        self.reward = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward = 0

        self.FLOOR_Y = self.HEIGHT - 40
        self.player_pos = [self.PLAYER_HORIZONTAL_POS, self.FLOOR_Y - self.PLAYER_SIZE]
        self.player_vel_y = 0
        self.on_ground = True
        self.jump_hold_frames = 0
        
        self.world_scroll_x = 0
        self.particles = []
        
        self.time_left = self.FPS * self.TIME_LIMIT_SECONDS
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), self.reward, True, False, self._get_info()

        # Unpack factorized action
        space_held = action[1] == 1

        self._update_player(space_held)
        self._update_particles()

        self.world_scroll_x += self.WORLD_SPEED
        self.steps += 1
        self.time_left -= 1
        self.score = int(self.world_scroll_x)

        self.reward = self._calculate_reward()
        terminated = self._check_termination()

        return (
            self._get_observation(),
            self.reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _generate_level(self):
        self.obstacles = []
        current_x = 400
        while current_x < self.LEVEL_LENGTH:
            progress_ratio = current_x / self.LEVEL_LENGTH
            gap_size = self.INITIAL_GAP_SIZE - (self.INITIAL_GAP_SIZE - self.FINAL_GAP_SIZE) * progress_ratio
            gap_size = max(self.FINAL_GAP_SIZE, gap_size)
            
            obstacle_width = self.np_random.integers(self.OBSTACLE_MIN_WIDTH, self.OBSTACLE_MAX_WIDTH + 1)
            obstacle_height = self.np_random.integers(self.OBSTACLE_MIN_HEIGHT, self.OBSTACLE_MAX_HEIGHT + 1)
            
            obstacle_rect = pygame.Rect(
                current_x + gap_size,
                self.FLOOR_Y - obstacle_height,
                obstacle_width,
                obstacle_height
            )
            self.obstacles.append(obstacle_rect)
            current_x += gap_size + obstacle_width
        
        self.finish_line_x = current_x + 200

    def _update_player(self, space_held):
        # Handle jumping
        if space_held and self.on_ground:
            # sound: player_jump.wav
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            self.jump_hold_frames = self.MAX_JUMP_HOLD_FRAMES
        elif space_held and self.jump_hold_frames > 0:
            self.player_vel_y -= self.JUMP_SUSTAIN_FORCE
            self.jump_hold_frames -= 1
        else:
            self.jump_hold_frames = 0

        # Apply gravity
        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y

        # Floor collision
        if self.player_pos[1] + self.PLAYER_SIZE >= self.FLOOR_Y:
            if not self.on_ground:
                # sound: player_land.wav
                self._create_landing_particles(10)
            self.player_pos[1] = self.FLOOR_Y - self.PLAYER_SIZE
            self.player_vel_y = 0
            self.on_ground = True

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # particle gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _create_landing_particles(self, count):
        for _ in range(count):
            self.particles.append({
                'pos': [self.player_pos[0] + self.PLAYER_SIZE / 2, self.FLOOR_Y],
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 0)],
                'lifespan': self.np_random.integers(15, 30),
                'color': random.choice([(200,200,220), (180,180,200), (220,220,240)])
            })

    def _calculate_reward(self):
        # Base reward for survival
        reward = 0.1

        if self._check_collision():
            # sound: player_hit.wav
            reward = -100.0
        elif self.world_scroll_x >= self.finish_line_x:
            # sound: level_win.wav
            reward = 100.0
        elif self.time_left <= 0:
            # sound: time_out.wav
            reward = -10.0
        
        return reward

    def _check_termination(self):
        if self._check_collision() or self.world_scroll_x >= self.finish_line_x or self.time_left <= 0:
            self.game_over = True
        return self.game_over

    def _check_collision(self):
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE, self.PLAYER_SIZE)
        for obs in self.obstacles:
            screen_obs_rect = obs.move(-self.world_scroll_x, 0)
            if player_rect.colliderect(screen_obs_rect):
                return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_world()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [
                self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_world(self):
        # Draw floor
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, (0, self.FLOOR_Y, self.WIDTH, self.HEIGHT - self.FLOOR_Y))
        
        # Draw obstacles
        for obs in self.obstacles:
            screen_rect = obs.move(-self.world_scroll_x, 0)
            if screen_rect.right > 0 and screen_rect.left < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, screen_rect, 3)

        # Draw finish line
        finish_screen_x = self.finish_line_x - self.world_scroll_x
        if 0 < finish_screen_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (finish_screen_x, 0), (finish_screen_x, self.FLOOR_Y), 5)

    def _render_player(self):
        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Glow effect
        glow_radius = int(self.PLAYER_SIZE * 1.2)
        glow_center = player_rect.center
        for i in range(glow_radius, 0, -2):
            alpha = 50 * (1 - i / glow_radius)
            pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], i, (*self.COLOR_PLAYER, alpha))
            
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p['lifespan'] / 6))
            pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], size)

    def _render_ui(self):
        def draw_text(text, font, color, pos, shadow_color=None, shadow_offset=(2,2)):
            text_surf = font.render(text, True, color)
            text_rect = text_surf.get_rect(topleft=pos)
            if shadow_color:
                shadow_surf = font.render(text, True, shadow_color)
                self.screen.blit(shadow_surf, (text_rect.x + shadow_offset[0], text_rect.y + shadow_offset[1]))
            self.screen.blit(text_surf, text_rect)

        # Score
        score_text = f"SCORE: {self.score}"
        draw_text(score_text, self.font_small, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        
        # Time
        time_display = f"TIME: {max(0, self.time_left / self.FPS):.1f}"
        time_surf = self.font_small.render(time_display, True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(topright=(self.WIDTH - 10, 10))
        shadow_surf = self.font_small.render(time_display, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(shadow_surf, (time_rect.x + 2, time_rect.y + 2))
        self.screen.blit(time_surf, time_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "distance_to_goal": max(0, self.finish_line_x - self.world_scroll_x)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to play the game
if __name__ == '__main__':
    env = GameEnv()
    
    # Use gymnasium's play utility
    try:
        from gymnasium.utils.play import play
        play(env, fps=60, keys_to_action={
            "w": np.array([0, 1, 0]),
            " ": np.array([0, 1, 0]),
            "s": np.array([0, 0, 0]), # No action for down
            "a": np.array([0, 0, 0]), # No action for left
            "d": np.array([0, 0, 0]), # No action for right
        })
    except ImportError:
        print("To play the game, run `pip install gymnasium[classic-control]`")
        # Manual fallback play loop
        obs, info = env.reset()
        done = False
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Minimalist Jumper")
        clock = pygame.time.Clock()
        
        while not done:
            action = np.array([0, 0, 0]) # Default no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
                action[1] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Reward: {reward}")

            # Render the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(env.FPS)
            
            if done:
                # Wait a bit before resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
                done = False

    env.close()