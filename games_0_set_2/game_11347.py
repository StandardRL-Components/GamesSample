import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


# Set up headless rendering for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade physics-based hockey game where the player launches comets to score goals.
    
    The game features:
    - A futuristic visual style with glowing elements and particle effects.
    - A physics engine simulating comet movement with gravity and friction.
    - A combo system rewarding consecutive goals.
    - Difficulty scaling through increasing gravity.
    - Unlockable comet types with different properties.
    - A simple AI opponent to compete against.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "An arcade physics-based hockey game where you launch comets to score goals against an AI opponent. "
        "Master the launch angle and power to win."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to aim and ↑↓ to adjust launch power. "
        "Press space to launch the comet and shift to cycle through unlocked comet types."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_BOARD = (20, 40, 80)
    COLOR_PLAYER_GOAL = (100, 255, 100)
    COLOR_OPPONENT_GOAL = (255, 100, 100)
    COLOR_PLAYER_PRIMARY = (50, 150, 255)
    COLOR_PLAYER_GLOW = (150, 200, 255)
    COLOR_OPPONENT_PRIMARY = (255, 50, 50)
    COLOR_OPPONENT_GLOW = (255, 150, 150)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TRAJECTORY = (255, 255, 255)
    
    # Game parameters
    GOAL_WIDTH = 120
    BOARD_PADDING = 20
    LAUNCH_AREA_Y = HEIGHT - 50
    MAX_STEPS = 1500
    WIN_SCORE = 10
    
    # Physics
    INITIAL_GRAVITY = 0.5
    GRAVITY_INCREMENT = 0.1
    LAUNCH_POWER_MIN = 2
    LAUNCH_POWER_MAX = 20
    LAUNCH_POWER_STEP = 0.4
    LAUNCH_ANGLE_STEP = 0.05  # Radians
    
    # Comet Types
    COMET_TYPES = [
        {'name': 'Standard', 'size': 10, 'friction': 0.99, 'color': COLOR_PLAYER_PRIMARY, 'glow': COLOR_PLAYER_GLOW},
        {'name': 'Heavy', 'size': 14, 'friction': 0.97, 'color': (200, 100, 255), 'glow': (230, 180, 255)},
        {'name': 'Fast', 'size': 7, 'friction': 0.995, 'color': (255, 255, 100), 'glow': (255, 255, 200)},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.board_rect = pygame.Rect(
            self.BOARD_PADDING, self.BOARD_PADDING,
            self.WIDTH - 2 * self.BOARD_PADDING, self.HEIGHT - 2 * self.BOARD_PADDING
        )
        self.player_goal_rect = pygame.Rect(
            (self.WIDTH - self.GOAL_WIDTH) // 2, self.HEIGHT - self.BOARD_PADDING,
            self.GOAL_WIDTH, self.BOARD_PADDING
        )
        self.opponent_goal_rect = pygame.Rect(
            (self.WIDTH - self.GOAL_WIDTH) // 2, 0,
            self.GOAL_WIDTH, self.BOARD_PADDING
        )
        
        # Persistent state across resets
        self.unlocked_comet_types = [True] + [False] * (len(self.COMET_TYPES) - 1)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.player_score = 0
        self.opponent_score = 0
        self.game_over = False
        
        self.gravity = self.INITIAL_GRAVITY
        self.launch_angle = -math.pi / 2
        self.launch_power = (self.LAUNCH_POWER_MIN + self.LAUNCH_POWER_MAX) / 2
        self.player_comet = None
        self.opponent_comet = None
        
        self.launch_cooldown = 0
        self.opponent_launch_cooldown = self.FPS * 2 # Initial delay
        
        self.current_comet_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.combo_count = 0
        self.last_comet_dist_to_goal = self.HEIGHT
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0

        # --- Update Game Logic ---
        self._handle_input(movement, space_held, shift_held)
        self._update_opponent_ai()
        self._update_physics()
        
        reward_info = self._check_events_and_calculate_rewards()
        reward += reward_info['reward']
        self.score += reward_info['reward']

        # --- Termination ---
        terminated = (self.player_score >= self.WIN_SCORE or
                      self.opponent_score >= self.WIN_SCORE or
                      self.steps >= self.MAX_STEPS)
        truncated = False
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.player_score >= self.WIN_SCORE:
                reward += 10
                self.score += 10
            elif self.opponent_score >= self.WIN_SCORE:
                reward -= 10
                self.score -= 10
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
        
    def _handle_input(self, movement, space_held, shift_held):
        is_aiming = self.player_comet is None and self.launch_cooldown <= 0

        if is_aiming:
            # Adjust angle
            if movement == 3: # Left
                self.launch_angle -= self.LAUNCH_ANGLE_STEP
            elif movement == 4: # Right
                self.launch_angle += self.LAUNCH_ANGLE_STEP
            self.launch_angle = max(-math.pi + 0.1, min(-0.1, self.launch_angle))

            # Adjust power
            if movement == 1: # Up
                self.launch_power += self.LAUNCH_POWER_STEP
            elif movement == 2: # Down
                self.launch_power -= self.LAUNCH_POWER_STEP
            self.launch_power = max(self.LAUNCH_POWER_MIN, min(self.LAUNCH_POWER_MAX, self.launch_power))

            # Cycle comet type
            if shift_held and not self.prev_shift_held:
                self.current_comet_idx = (self.current_comet_idx + 1) % len(self.COMET_TYPES)
                while not self.unlocked_comet_types[self.current_comet_idx]:
                    self.current_comet_idx = (self.current_comet_idx + 1) % len(self.COMET_TYPES)

            # Launch
            if space_held and not self.prev_space_held:
                comet_type = self.COMET_TYPES[self.current_comet_idx]
                vel_x = self.launch_power * math.cos(self.launch_angle)
                vel_y = self.launch_power * math.sin(self.launch_angle)
                self.player_comet = {
                    'pos': np.array([self.WIDTH / 2, self.LAUNCH_AREA_Y], dtype=float),
                    'vel': np.array([vel_x, vel_y], dtype=float),
                    'trail': [],
                    **comet_type
                }
                self.launch_cooldown = self.FPS // 2 # 0.5s cooldown
                self.last_comet_dist_to_goal = np.linalg.norm(self.player_comet['pos'] - self.opponent_goal_rect.center)
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        if self.launch_cooldown > 0:
            self.launch_cooldown -= 1

    def _update_opponent_ai(self):
        if self.opponent_launch_cooldown > 0:
            self.opponent_launch_cooldown -= 1
        
        if self.opponent_comet is None and self.opponent_launch_cooldown <= 0:
            target_x = self.player_goal_rect.centerx + self.np_random.uniform(-self.GOAL_WIDTH/3, self.GOAL_WIDTH/3)
            target_y = self.player_goal_rect.centery
            launch_pos = np.array([self.WIDTH / 2, self.BOARD_PADDING + 20], dtype=float)
            
            angle = math.atan2(target_y - launch_pos[1], target_x - launch_pos[0])
            power = self.np_random.uniform(10, 15)
            
            vel_x = power * math.cos(angle)
            vel_y = power * math.sin(angle)
            
            self.opponent_comet = {
                'pos': launch_pos,
                'vel': np.array([vel_x, vel_y], dtype=float),
                'trail': [],
                'size': 10,
                'friction': 0.99,
                'color': self.COLOR_OPPONENT_PRIMARY,
                'glow': self.COLOR_OPPONENT_GLOW
            }
            self.opponent_launch_cooldown = self.FPS * self.np_random.uniform(2, 4)

    def _update_physics(self):
        # Update comets
        for comet in [self.player_comet, self.opponent_comet]:
            if comet:
                comet['trail'].append(comet['pos'].copy())
                if len(comet['trail']) > 20:
                    comet['trail'].pop(0)

                comet['vel'][1] += self.gravity / self.FPS # Apply gravity
                comet['vel'] *= comet['friction'] # Apply friction
                comet['pos'] += comet['vel']

                # Wall collisions
                if comet['pos'][0] < self.board_rect.left + comet['size'] or comet['pos'][0] > self.board_rect.right - comet['size']:
                    comet['vel'][0] *= -0.9
                    comet['pos'][0] = np.clip(comet['pos'][0], self.board_rect.left + comet['size'], self.board_rect.right - comet['size'])
                if comet['pos'][1] < self.board_rect.top + comet['size'] or comet['pos'][1] > self.board_rect.bottom - comet['size']:
                    # Reset comet if it hits top/bottom walls outside of goal areas
                    if (comet is self.player_comet and not self.opponent_goal_rect.collidepoint(comet['pos'])) or \
                       (comet is self.opponent_comet and not self.player_goal_rect.collidepoint(comet['pos'])):
                        if comet is self.player_comet: self.player_comet = None
                        else: self.opponent_comet = None
        
        # Update particles
        self.particles = [(p[0], p[1] - 1, p[2] - 1) for p in self.particles if p[2] > 0]

    def _check_events_and_calculate_rewards(self):
        reward = 0
        
        # Player comet events
        if self.player_comet:
            # Continuous reward for getting closer to goal
            dist_to_goal = np.linalg.norm(self.player_comet['pos'] - self.opponent_goal_rect.center)
            if dist_to_goal < self.last_comet_dist_to_goal:
                reward += 0.01
            self.last_comet_dist_to_goal = dist_to_goal
            
            # Opponent goal
            if self.opponent_goal_rect.collidepoint(self.player_comet['pos']):
                self._create_explosion(self.player_comet['pos'], self.COLOR_PLAYER_GOAL)
                self.player_score += 1
                self.combo_count += 1
                reward += 1
                if self.combo_count == 2: reward += 0.5
                if self.combo_count > 2: reward += 1
                self._update_difficulty_and_unlocks()
                self.player_comet = None
            
            # Own goal
            elif self.player_goal_rect.collidepoint(self.player_comet['pos']):
                self._create_explosion(self.player_comet['pos'], self.COLOR_OPPONENT_GOAL)
                self.opponent_score += 1
                self.combo_count = 0
                reward -= 1
                self._update_difficulty_and_unlocks()
                self.player_comet = None

        # Opponent comet events
        if self.opponent_comet:
            if self.player_goal_rect.collidepoint(self.opponent_comet['pos']):
                self._create_explosion(self.opponent_comet['pos'], self.COLOR_OPPONENT_GOAL)
                self.opponent_score += 1
                self.combo_count = 0
                reward -= 1
                self._update_difficulty_and_unlocks()
                self.opponent_comet = None
                
        return {'reward': reward}

    def _update_difficulty_and_unlocks(self):
        total_goals = self.player_score + self.opponent_score
        self.gravity = self.INITIAL_GRAVITY + (total_goals // 2) * self.GRAVITY_INCREMENT

        # Unlock new comets based on player score
        if self.player_score >= 3 and not self.unlocked_comet_types[1]:
            self.unlocked_comet_types[1] = True
        if self.player_score >= 6 and not self.unlocked_comet_types[2]:
            self.unlocked_comet_types[2] = True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Board
        pygame.draw.rect(self.screen, self.COLOR_BOARD, self.board_rect, border_radius=10)
        
        # Goals
        pygame.draw.rect(self.screen, self.COLOR_OPPONENT_GOAL, self.opponent_goal_rect, border_top_left_radius=5, border_top_right_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GOAL, self.player_goal_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)
        
        # Particles
        for p in self.particles:
            pos, color, life = p
            pygame.draw.circle(self.screen, color, pos, life)

        # Comets
        for comet in [self.player_comet, self.opponent_comet]:
            if comet:
                self._render_comet_trail(comet)
                self._render_glowing_circle(comet['pos'], comet['size'], comet['color'], comet['glow'])
        
        # Trajectory prediction and aiming ghost
        if self.player_comet is None and self.launch_cooldown <= 0:
            self._render_trajectory_prediction()
            comet_type = self.COMET_TYPES[self.current_comet_idx]
            self._render_glowing_circle(
                (self.WIDTH/2, self.LAUNCH_AREA_Y), comet_type['size'], 
                comet_type['color'], comet_type['glow'], alpha=150
            )

    def _render_ui(self):
        # Score
        score_text = f"{self.player_score} - {self.opponent_score}"
        text_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH / 2 - text_surf.get_width() / 2, 25))

        # Combo
        if self.combo_count > 1:
            combo_text = f"COMBO x{self.combo_count}"
            text_surf = self.font_small.render(combo_text, True, self.COLOR_PLAYER_GOAL)
            self.screen.blit(text_surf, (self.WIDTH / 2 - text_surf.get_width() / 2, 55))

        # Power Meter
        if self.player_comet is None and self.launch_cooldown <= 0:
            meter_x = self.board_rect.left - 15
            meter_height = 100
            meter_y = self.HEIGHT - self.BOARD_PADDING - meter_height
            power_ratio = (self.launch_power - self.LAUNCH_POWER_MIN) / (self.LAUNCH_POWER_MAX - self.LAUNCH_POWER_MIN)
            
            pygame.draw.rect(self.screen, self.COLOR_BOARD, (meter_x, meter_y, 10, meter_height))
            fill_height = int(meter_height * power_ratio)
            fill_color = (255, 255 - 200 * power_ratio, 0)
            pygame.draw.rect(self.screen, fill_color, (meter_x, meter_y + meter_height - fill_height, 10, fill_height))
            
            # Comet Type Display
            comet_name = self.COMET_TYPES[self.current_comet_idx]['name']
            type_text = self.font_small.render(comet_name, True, self.COLOR_TEXT)
            self.screen.blit(type_text, (self.board_rect.right + 10, self.HEIGHT - self.BOARD_PADDING - 20))

        # Game Over Text
        if self.game_over:
            if self.player_score >= self.WIN_SCORE:
                end_text = "YOU WIN"
                color = self.COLOR_PLAYER_GOAL
            else:
                end_text = "YOU LOSE"
                color = self.COLOR_OPPONENT_GOAL
            text_surf = self.font_large.render(end_text, True, color)
            self.screen.blit(text_surf, (self.WIDTH/2 - text_surf.get_width()/2, self.HEIGHT/2 - text_surf.get_height()/2))
            
    def _render_glowing_circle(self, pos, radius, color, glow_color, alpha=255):
        int_pos = (int(pos[0]), int(pos[1]))
        glow_radius = int(radius * 1.8)
        
        # Using gfxdraw for anti-aliased, alpha-blended circles
        for i in range(glow_radius, int(radius-1), -1):
            r, g, b = glow_color
            if i > radius:
                # Fade out glow
                a = int(alpha * (1 - (i - radius) / (glow_radius - radius))**2 * 0.5)
            else:
                # Core color
                r_core, g_core, b_core = color
                lerp = (radius - i) / radius
                r = int(r_core * lerp + r * (1-lerp))
                g = int(g_core * lerp + g * (1-lerp))
                b = int(b_core * lerp + b * (1-lerp))
                a = alpha
            
            if a > 0:
                pygame.gfxdraw.filled_circle(self.screen, int_pos[0], int_pos[1], i, (r, g, b, a))

    def _render_comet_trail(self, comet):
        for i, pos in enumerate(comet['trail']):
            alpha = int(200 * (i / len(comet['trail'])))
            radius = int(comet['size'] * (i / len(comet['trail'])) * 0.5)
            if radius > 0 and alpha > 0:
                r, g, b = comet['glow']
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, (r, g, b, alpha))

    def _render_trajectory_prediction(self):
        comet_type = self.COMET_TYPES[self.current_comet_idx]
        pos = np.array([self.WIDTH / 2, self.LAUNCH_AREA_Y], dtype=float)
        vel = np.array([
            self.launch_power * math.cos(self.launch_angle),
            self.launch_power * math.sin(self.launch_angle)
        ], dtype=float)
        
        for i in range(50):
            vel[1] += self.gravity / self.FPS
            vel *= comet_type['friction']
            pos += vel
            if i % 4 == 0:
                alpha = 200 * (1 - i / 50)
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 2, (*self.COLOR_TRAJECTORY, int(alpha)))

    def _create_explosion(self, pos, color, num_particles=50):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            life = self.np_random.integers(10, 20)
            self.particles.append((pos, color, life))
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
            "combo": self.combo_count,
            "gravity": self.gravity
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # ARROWS: Aim and adjust power
    # SPACE: Launch
    # SHIFT: Cycle comet type
    # Q: Quit
    
    action = [0, 0, 0] # [movement, space, shift]
    
    # The main display needs to be initialized for manual play
    display = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Comet Hockey")
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 0

        keys = pygame.key.get_pressed()
        action[0] = 0 # No movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        # Reset action keys that are not held down
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

    print(f"Game Over. Final Info: {info}")
    env.close()