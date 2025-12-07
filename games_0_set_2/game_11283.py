import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:55:42.867583
# Source Brief: brief_01283.md
# Brief Index: 1283
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a skill-based arcade game.
    The player controls the launch angle of bouncing balls to hit targets.
    Hitting targets increases the score and the ball's speed.
    Unlocking new, faster balls is key to achieving a high score within the time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch bouncing balls to hit targets and score points. "
        "Unlock faster balls to achieve a high score before time runs out."
    )
    user_guide = "Controls: ←→ to aim, space to launch, and shift to select ball type."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAY_AREA_PADDING = 10
    FPS = 60
    GAME_DURATION_SECONDS = 60

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_BORDER = (100, 110, 130)
    COLOR_TARGET = (255, 80, 80)
    COLOR_TARGET_GLOW = (255, 80, 80, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 40, 60, 180)
    COLOR_UI_HIGHLIGHT = (255, 255, 0)
    COLOR_LOCKED = (80, 80, 80)
    
    BALL_SPECS = [
        {'color': (0, 255, 150), 'radius': 12, 'base_speed': 4.0}, # Green
        {'color': (0, 150, 255), 'radius': 10, 'base_speed': 6.0}, # Blue
        {'color': (255, 255, 0), 'radius': 8,  'base_speed': 8.0}, # Yellow
        {'color': (255, 150, 0), 'radius': 6,  'base_speed': 10.0} # Orange
    ]
    BALL_UNLOCK_THRESHOLD = 10
    NUM_TARGETS = 3
    TARGET_RADIUS = 20
    TARGET_RESPAWN_TIME = 0.5 * FPS # in steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # Play area definition
        self.play_area = pygame.Rect(
            self.PLAY_AREA_PADDING,
            self.PLAY_AREA_PADDING,
            self.SCREEN_WIDTH - 2 * self.PLAY_AREA_PADDING,
            self.SCREEN_HEIGHT - 2 * self.PLAY_AREA_PADDING - 60 # Space for UI
        )
        
        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.launcher_pos = pygame.Vector2(0, 0)
        self.launch_angle = 0.0
        self.balls = []
        self.targets = []
        self.particles = []
        self.selected_ball_type = 0
        self.unlocked_ball_types = []
        self.ball_hit_counts = []
        self.ball_current_speeds = []
        self.last_space_held = False
        self.last_shift_held = False
        self.launch_cooldown = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.GAME_DURATION_SECONDS * self.FPS
        
        self.launcher_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 45)
        self.launch_angle = -math.pi / 2 # Pointing straight up

        self.balls = []
        self.particles = []
        
        self.selected_ball_type = 0
        self.unlocked_ball_types = [True] + [False] * (len(self.BALL_SPECS) - 1)
        self.ball_hit_counts = [0] * len(self.BALL_SPECS)
        self.ball_current_speeds = [spec['base_speed'] for spec in self.BALL_SPECS]

        self.last_space_held = False
        self.last_shift_held = False
        self.launch_cooldown = 0

        self.targets = []
        for _ in range(self.NUM_TARGETS):
            self._spawn_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- 1. Handle Input and State Changes ---
        self._handle_input(action)
        
        self.steps += 1
        self.time_remaining -= 1
        if self.launch_cooldown > 0:
            self.launch_cooldown -= 1

        # --- 2. Update Game Entities ---
        self._update_balls()
        self._update_particles()
        reward += self._update_targets_and_collisions()
        
        # --- 3. Check for Unlocks ---
        reward += self._check_for_unlocks()

        # --- 4. Check for Termination ---
        terminated = self.time_remaining <= 0 or self.score >= 500
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= 500:
                reward += 100 # Victory reward
            else:
                reward -= 100 # Time-out penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Adjust launch angle
        angle_speed = 0.05
        if movement == 3: # Left
            self.launch_angle -= angle_speed
        elif movement == 4: # Right
            self.launch_angle += angle_speed
        # Keep angle within a reasonable range to prevent shooting backwards
        self.launch_angle = max(-math.pi + 0.1, min(-0.1, self.launch_angle))

        # Launch ball on rising edge of space key
        space_pressed = space_held and not self.last_space_held
        if space_pressed and self.launch_cooldown == 0:
            self._launch_ball()
            self.launch_cooldown = 10 # 1/6th of a second cooldown
            # Sound effect placeholder: # sfx_launch.play()

        # Cycle ball type on rising edge of shift key
        shift_pressed = shift_held and not self.last_shift_held
        if shift_pressed:
            self._cycle_ball_type()
            # Sound effect placeholder: # sfx_select.play()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _launch_ball(self):
        spec = self.BALL_SPECS[self.selected_ball_type]
        speed = self.ball_current_speeds[self.selected_ball_type]
        
        velocity = pygame.Vector2(math.cos(self.launch_angle), math.sin(self.launch_angle)) * speed
        
        ball = {
            'pos': self.launcher_pos.copy(),
            'vel': velocity,
            'radius': spec['radius'],
            'color': spec['color'],
            'type': self.selected_ball_type
        }
        self.balls.append(ball)

    def _cycle_ball_type(self):
        self.selected_ball_type = (self.selected_ball_type + 1) % len(self.BALL_SPECS)
        # Skip locked balls
        while not self.unlocked_ball_types[self.selected_ball_type]:
            self.selected_ball_type = (self.selected_ball_type + 1) % len(self.BALL_SPECS)

    def _update_balls(self):
        for ball in self.balls:
            ball['pos'] += ball['vel']
            # Wall collisions
            if ball['pos'].x - ball['radius'] < self.play_area.left or \
               ball['pos'].x + ball['radius'] > self.play_area.right:
                ball['vel'].x *= -1
                ball['pos'].x = np.clip(ball['pos'].x, self.play_area.left + ball['radius'], self.play_area.right - ball['radius'])
                # Sound effect placeholder: # sfx_bounce.play()
            if ball['pos'].y - ball['radius'] < self.play_area.top or \
               ball['pos'].y + ball['radius'] > self.play_area.bottom:
                ball['vel'].y *= -1
                ball['pos'].y = np.clip(ball['pos'].y, self.play_area.top + ball['radius'], self.play_area.bottom - ball['radius'])
                # Sound effect placeholder: # sfx_bounce.play()
        
        # Remove balls that go out of bounds (should not happen with bouncing, but good practice)
        self.balls = [b for b in self.balls if self.play_area.collidepoint(b['pos'])]

    def _update_targets_and_collisions(self):
        hit_reward = 0
        for target in self.targets:
            if target['respawn_timer'] > 0:
                target['respawn_timer'] -= 1
                if target['respawn_timer'] == 0:
                    self._respawn_target(target)
                continue

            for ball in self.balls:
                distance = target['pos'].distance_to(ball['pos'])
                if distance < self.TARGET_RADIUS + ball['radius']:
                    # Collision occurred
                    self.score += 10
                    hit_reward += 0.1
                    self.ball_hit_counts[ball['type']] += 1
                    self.ball_current_speeds[ball['type']] *= 1.05 # 5% speed increase
                    
                    self._create_particles(target['pos'], self.COLOR_TARGET, 20)
                    target['respawn_timer'] = self.TARGET_RESPAWN_TIME
                    
                    # Bounce ball off target
                    normal = (ball['pos'] - target['pos']).normalize()
                    ball['vel'].reflect_ip(normal)
                    # Sound effect placeholder: # sfx_hit_target.play()
                    break # Target can only be hit once per frame
        return hit_reward

    def _check_for_unlocks(self):
        unlock_reward = 0
        for i in range(len(self.BALL_SPECS) - 1):
            if self.unlocked_ball_types[i] and not self.unlocked_ball_types[i+1]:
                if self.ball_hit_counts[i] >= self.BALL_UNLOCK_THRESHOLD:
                    self.unlocked_ball_types[i+1] = True
                    unlock_reward += 1.0
                    self._create_particles(self.launcher_pos, self.BALL_SPECS[i+1]['color'], 50)
                    # Sound effect placeholder: # sfx_unlock.play()
        return unlock_reward

    def _spawn_target(self):
        # Find a valid position away from other targets
        while True:
            pos = pygame.Vector2(
                random.uniform(self.play_area.left + self.TARGET_RADIUS, self.play_area.right - self.TARGET_RADIUS),
                random.uniform(self.play_area.top + self.TARGET_RADIUS, self.play_area.bottom - self.TARGET_RADIUS * 3) # Keep away from launcher area
            )
            if all(pos.distance_to(t['pos']) > self.TARGET_RADIUS * 3 for t in self.targets):
                break
        
        self.targets.append({
            'pos': pos,
            'respawn_timer': 0
        })

    def _respawn_target(self, target):
        while True:
            pos = pygame.Vector2(
                random.uniform(self.play_area.left + self.TARGET_RADIUS, self.play_area.right - self.TARGET_RADIUS),
                random.uniform(self.play_area.top + self.TARGET_RADIUS, self.play_area.bottom - self.TARGET_RADIUS * 3)
            )
            # Check distance to other targets (excluding itself)
            if all(pos.distance_to(t['pos']) > self.TARGET_RADIUS * 3 for t in self.targets if t is not target):
                target['pos'] = pos
                break

    def _create_particles(self, position, color, count):
        for _ in range(count):
            self.particles.append({
                'pos': position.copy(),
                'vel': pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3)),
                'lifetime': random.randint(20, 40),
                'color': color,
                'radius': random.uniform(1, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Play area border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, self.play_area, 2)

        # Targets
        for target in self.targets:
            if target['respawn_timer'] > 0:
                # Fade out effect
                alpha = int(255 * (target['respawn_timer'] / self.TARGET_RESPAWN_TIME))
                color = (*self.COLOR_TARGET, alpha)
                temp_surf = pygame.Surface((self.TARGET_RADIUS*2, self.TARGET_RADIUS*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (self.TARGET_RADIUS, self.TARGET_RADIUS), self.TARGET_RADIUS)
                self.screen.blit(temp_surf, (int(target['pos'].x - self.TARGET_RADIUS), int(target['pos'].y - self.TARGET_RADIUS)))
            else:
                # Glow effect
                pygame.gfxdraw.filled_circle(self.screen, int(target['pos'].x), int(target['pos'].y), self.TARGET_RADIUS + 5, self.COLOR_TARGET_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, int(target['pos'].x), int(target['pos'].y), self.TARGET_RADIUS, self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, int(target['pos'].x), int(target['pos'].y), self.TARGET_RADIUS, self.COLOR_TARGET)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 40))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

        # Balls
        for ball in self.balls:
            pos = (int(ball['pos'].x), int(ball['pos'].y))
            # Glow
            glow_color = (*ball['color'], 60)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(ball['radius'] * 1.5), glow_color)
            # Ball
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], ball['radius'], ball['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], ball['radius'], ball['color'])
            
        # Launcher
        end_pos = self.launcher_pos + pygame.Vector2(math.cos(self.launch_angle), math.sin(self.launch_angle)) * 40
        pygame.draw.line(self.screen, self.COLOR_TEXT, self.launcher_pos, end_pos, 3)
        pygame.draw.circle(self.screen, self.COLOR_TEXT, (int(self.launcher_pos.x), int(self.launcher_pos.y)), 8)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, self.SCREEN_HEIGHT - 40))

        # Timer
        time_str = f"TIME: {self.time_remaining // self.FPS:02d}"
        timer_text = self.font_large.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 15, 10))

        # Ball Selector
        ui_bar_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 60, self.SCREEN_WIDTH, 60)
        
        total_width = len(self.BALL_SPECS) * 50
        start_x = self.SCREEN_WIDTH / 2 - total_width / 2

        for i, spec in enumerate(self.BALL_SPECS):
            center_x = start_x + i * 50 + 25
            center_y = self.SCREEN_HEIGHT - 35
            
            if self.unlocked_ball_types[i]:
                color = spec['color']
                if i == self.selected_ball_type:
                    # Highlight selected
                    pygame.draw.rect(self.screen, self.COLOR_UI_HIGHLIGHT, (center_x - 20, center_y - 20, 40, 40), 2, border_radius=5)
            else:
                color = self.COLOR_LOCKED

            pygame.draw.circle(self.screen, color, (int(center_x), int(center_y)), spec['radius'])
            
            # Draw hit count for unlocked balls
            if self.unlocked_ball_types[i] and i < len(self.BALL_SPECS) - 1 and not self.unlocked_ball_types[i+1]:
                hit_count_str = f"{self.ball_hit_counts[i]}/{self.BALL_UNLOCK_THRESHOLD}"
                hit_text = self.font_small.render(hit_count_str, True, self.COLOR_TEXT)
                self.screen.blit(hit_text, (center_x - hit_text.get_width() / 2, center_y + 12))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining // self.FPS,
            "unlocked_balls": sum(self.unlocked_ball_types)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and visualize the game
if __name__ == "__main__":
    # This block will not run in the hosted environment but is useful for local testing.
    # We need to unset the dummy video driver to see the window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a Pygame window
    pygame.display.set_caption("Bouncer Game")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # ARROWS: Change angle
    # SPACE: Launch ball
    # SHIFT: Cycle ball type
    
    action = np.array([0, 0, 0]) # [movement, space, shift]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_LEFT]:
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
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()