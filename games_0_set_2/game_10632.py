import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    GameEnv: Master precision and timing to control two robotic arms simultaneously,
    welding parts against the clock for a high score.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control two robotic arms to weld parts on a moving conveyor belt. "
        "Time your welds perfectly for score multipliers and aim for a high score before the clock runs out."
    )
    user_guide = (
        "Controls: Use ←→ to move the left arm. Hold Shift and use ←→ to move the right arm. "
        "Press Space to weld."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_TIME_SECONDS = 60
    MAX_STEPS = MAX_TIME_SECONDS * FPS
    WIN_SCORE = 300

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_CONVEYOR = (25, 30, 40)
    COLOR_ARM_BODY = (40, 120, 200)
    COLOR_ARM_GLOW = (40, 120, 200, 50)
    COLOR_ARM_TIP = (150, 220, 255)
    COLOR_PART = (100, 110, 120)
    COLOR_PART_WELD_POINT = (50, 200, 100)
    COLOR_WELD_PERFECT = (255, 220, 50)
    COLOR_WELD_IMPERFECT = (255, 80, 50)
    COLOR_WELD_IN_PROGRESS = (240, 240, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_MULTIPLIER = (255, 220, 50)
    
    # Game Parameters
    ARM_SPEED = 8
    ARM_WIDTH = 20
    ARM_HEIGHT = 150
    WELDING_ZONE_Y = 280
    PART_WIDTH = 60
    PART_HEIGHT = 20
    PART_SPAWN_RATE = 2 * FPS # Every 2 seconds
    PART_SPEED = 1.0
    WELD_TOLERANCE_PERFECT = 5
    WELD_TOLERANCE_IMPERFECT = 20
    MULTIPLIER_THRESHOLDS = [0, 2, 5, 10] # Consecutive perfect welds for x2, x3, x4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # State variables initialized in reset()
        self.steps = None
        self.score = None
        self.time_remaining = None
        self.game_over = None
        self.win = None
        self.arm_left_pos_x = None
        self.arm_right_pos_x = None
        self.parts = None
        self.part_spawn_timer = None
        self.particles = None
        self.prev_space_held = None
        self.event_reward = None
        self.multiplier = None
        self.consecutive_perfect_welds = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.win = False

        self.arm_left_pos_x = self.SCREEN_WIDTH * 0.25
        self.arm_right_pos_x = self.SCREEN_WIDTH * 0.75
        
        self.parts = []
        self.part_spawn_timer = self.PART_SPAWN_RATE // 2
        for i in range(3):
            self._spawn_part(initial=True, y_offset=i * self.SCREEN_HEIGHT/3)

        self.particles = []
        self.prev_space_held = False
        self.event_reward = 0
        self.multiplier = 1
        self.consecutive_perfect_welds = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        self.event_reward = 0

        self._handle_input(action)
        self._update_game_state()
        
        reward = self._calculate_reward()
        terminated = self._check_termination()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        move_cmd = 0
        if movement == 3: move_cmd = -1 # Left
        elif movement == 4: move_cmd = 1 # Right

        if move_cmd != 0:
            if shift_held:
                self.arm_right_pos_x += move_cmd * self.ARM_SPEED
            else:
                self.arm_left_pos_x += move_cmd * self.ARM_SPEED

        # Clamp arm positions
        arm_half_width = self.ARM_WIDTH // 2
        self.arm_left_pos_x = np.clip(self.arm_left_pos_x, arm_half_width, self.SCREEN_WIDTH - arm_half_width)
        self.arm_right_pos_x = np.clip(self.arm_right_pos_x, arm_half_width, self.SCREEN_WIDTH - arm_half_width)

        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            self._try_weld()
        
        self.prev_space_held = space_held

    def _try_weld(self):
        weldable_part = self._get_weldable_part()
        if weldable_part is None:
            # SFX: Weld_Miss.wav
            return

        part_x, part_y, part_w = weldable_part['x'], weldable_part['y'], weldable_part['w']
        
        left_weld_target = part_x - part_w / 2
        right_weld_target = part_x + part_w / 2

        dist_left = abs(self.arm_left_pos_x - left_weld_target)
        dist_right = abs(self.arm_right_pos_x - right_weld_target)
        
        total_dist = dist_left + dist_right

        if total_dist <= self.WELD_TOLERANCE_PERFECT:
            # SFX: Weld_Perfect.wav
            quality = 'perfect'
            base_points = 10
            self.consecutive_perfect_welds += 1
            self.event_reward += 5
        elif total_dist <= self.WELD_TOLERANCE_IMPERFECT:
            # SFX: Weld_Imperfect.wav
            quality = 'imperfect'
            base_points = 2
            self.consecutive_perfect_welds = 0
            self.event_reward -= 1
        else:
            return # Weld failed, arms not close enough

        # Update multiplier
        if self.consecutive_perfect_welds >= self.MULTIPLIER_THRESHOLDS[3]:
            self.multiplier = 4
        elif self.consecutive_perfect_welds >= self.MULTIPLIER_THRESHOLDS[2]:
            self.multiplier = 3
        elif self.consecutive_perfect_welds >= self.MULTIPLIER_THRESHOLDS[1]:
            self.multiplier = 2
        else:
            self.multiplier = 1

        self.score += base_points * self.multiplier
        
        self._create_sparks(part_x, self.WELDING_ZONE_Y, quality)
        self.parts.remove(weldable_part)

    def _get_weldable_part(self):
        best_part = None
        min_dist = float('inf')
        for part in self.parts:
            dist_y = abs(part['y'] - self.WELDING_ZONE_Y)
            if dist_y < min_dist and dist_y < self.PART_HEIGHT * 2:
                min_dist = dist_y
                best_part = part
        return best_part

    def _update_game_state(self):
        # Update parts
        self.part_spawn_timer -= 1
        if self.part_spawn_timer <= 0:
            self._spawn_part()
            self.part_spawn_timer = self.PART_SPAWN_RATE

        for part in self.parts:
            part['y'] += self.PART_SPEED
        
        self.parts = [p for p in self.parts if p['y'] < self.SCREEN_HEIGHT + self.PART_HEIGHT]

        # Update particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _spawn_part(self, initial=False, y_offset=0):
        part_x = self.np_random.uniform(self.SCREEN_WIDTH * 0.3, self.SCREEN_WIDTH * 0.7)
        part_y = -self.PART_HEIGHT - y_offset
        if initial:
            part_y = y_offset
        self.parts.append({'x': part_x, 'y': part_y, 'w': self.PART_WIDTH})

    def _create_sparks(self, x, y, quality):
        # SFX: Sparks.wav
        if quality == 'perfect':
            color = self.COLOR_WELD_PERFECT
            count = 40
        else: # imperfect
            color = self.COLOR_WELD_IMPERFECT
            count = 20
        
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'color': color,
                'size': self.np_random.uniform(1, 3)
            })

    def _calculate_reward(self):
        reward = self.event_reward

        # Alignment reward
        part = self._get_weldable_part()
        if part:
            left_target = part['x'] - part['w'] / 2
            right_target = part['x'] + part['w'] / 2
            dist_left = abs(self.arm_left_pos_x - left_target)
            dist_right = abs(self.arm_right_pos_x - right_target)
            if dist_left + dist_right < self.WELD_TOLERANCE_IMPERFECT:
                reward += 0.1
        
        if self.game_over and self.win:
            reward += 100

        return reward

    def _check_termination(self):
        if self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
        return self.game_over

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_parts()
        self._render_arms()
        self._render_effects()
        self._render_ui()

    def _render_background(self):
        # Conveyor belt
        belt_rect = pygame.Rect(self.SCREEN_WIDTH * 0.2, 0, self.SCREEN_WIDTH * 0.6, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_CONVEYOR, belt_rect)

        # Welding zone line
        pygame.draw.line(self.screen, self.COLOR_ARM_TIP, (belt_rect.left, self.WELDING_ZONE_Y), (belt_rect.right, self.WELDING_ZONE_Y), 2)
        
        # Conveyor lines for motion effect
        line_y_start = (self.steps * self.PART_SPEED) % 40
        for i in range(self.SCREEN_HEIGHT // 40 + 2):
            y = line_y_start + i * 40
            pygame.draw.line(self.screen, self.COLOR_BG, (belt_rect.left, y), (belt_rect.right, y), 1)

    def _render_parts(self):
        weldable_part = self._get_weldable_part()
        for part in self.parts:
            part_rect = pygame.Rect(part['x'] - part['w'] / 2, part['y'] - self.PART_HEIGHT / 2, part['w'], self.PART_HEIGHT)
            is_weldable = part is weldable_part
            
            part_color = (130, 140, 150) if is_weldable else self.COLOR_PART
            pygame.draw.rect(self.screen, part_color, part_rect, border_radius=3)
            
            # Weld points
            point_radius = 4 if is_weldable else 3
            point_color = self.COLOR_PART_WELD_POINT
            pygame.gfxdraw.filled_circle(self.screen, int(part_rect.left), int(part_rect.centery), point_radius, point_color)
            pygame.gfxdraw.filled_circle(self.screen, int(part_rect.right), int(part_rect.centery), point_radius, point_color)

    def _render_arms(self):
        # Helper to draw an arm with glow
        def draw_arm(x, height, width):
            # Glow
            glow_rect = pygame.Rect(x - width, 0, width * 2, height)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_ARM_GLOW, (0, 0, glow_rect.width, glow_rect.height), border_radius=8)
            self.screen.blit(glow_surf, glow_rect.topleft)
            # Body
            body_rect = pygame.Rect(x - width / 2, 0, width, height)
            pygame.draw.rect(self.screen, self.COLOR_ARM_BODY, body_rect, border_radius=5)
            # Tip
            pygame.draw.circle(self.screen, self.COLOR_ARM_TIP, (int(x), height), 8)
            pygame.draw.circle(self.screen, self.COLOR_WELD_IN_PROGRESS, (int(x), height), 4)

        draw_arm(self.arm_left_pos_x, self.WELDING_ZONE_Y, self.ARM_WIDTH)
        draw_arm(self.arm_right_pos_x, self.WELDING_ZONE_Y, self.ARM_WIDTH)

    def _render_effects(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            # Use gfxdraw for antialiasing
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)

    def _render_ui(self):
        # Score
        score_text = self.font_big.render(f"{self.score:04d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_str = f"{self.time_remaining / self.FPS:.1f}"
        time_text = self.font_big.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 20, 10))
        
        # Multiplier
        if self.multiplier > 1:
            mult_text = self.font_big.render(f"x{self.multiplier}", True, self.COLOR_MULTIPLIER)
            self.screen.blit(mult_text, (self.SCREEN_WIDTH / 2 - mult_text.get_width() / 2, self.SCREEN_HEIGHT - 50))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "TIME'S UP"
            color = self.COLOR_WELD_PERFECT if self.win else self.COLOR_WELD_IMPERFECT
            
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining / self.FPS,
            "multiplier": self.multiplier,
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # --- Manual Play ---
    # To control:
    # A/D: Move left arm
    # Left/Right Arrow + Shift: Move right arm
    # Space: Weld
    
    # Pygame setup for rendering
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Welding Arms")
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        keys = pygame.key.get_pressed()
        
        # Left arm movement
        if keys[pygame.K_a]:
            action[0] = 3 # left
        elif keys[pygame.K_d]:
            action[0] = 4 # right
        
        # Right arm movement (requires shift)
        if keys[pygame.K_LEFT] and keys[pygame.K_LSHIFT]:
            action[0] = 3
            action[2] = 1
        elif keys[pygame.K_RIGHT] and keys[pygame.K_LSHIFT]:
            action[0] = 4
            action[2] = 1

        # Welding
        if keys[pygame.K_SPACE]:
            action[1] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for reset
        
        clock.tick(env.FPS)

    env.close()