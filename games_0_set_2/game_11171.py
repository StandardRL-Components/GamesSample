import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for visual effects
class Particle:
    def __init__(self, x, y, vx, vy, size, color, lifespan):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.color = color
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravity
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            temp_surf = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.color + (alpha,), (int(self.size), int(self.size)), int(self.size))
            surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)

class FloatingText:
    def __init__(self, x, y, text, font, color, lifespan):
        self.x = x
        self.y = y
        self.text = text
        self.font = font
        self.color = color
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self):
        self.y -= 1
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            text_surface = self.font.render(self.text, True, self.color)
            text_surface.set_alpha(alpha)
            surface.blit(text_surface, (int(self.x - text_surface.get_width() / 2), int(self.y)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Spin a prize wheel to score points against the clock. Land on gold segments consecutively to trigger a combo multiplier for a high score."
    )
    user_guide = (
        "Controls: Press space to stop the spinning wheel and score points."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    WIN_SCORE = 500
    TOTAL_TIME_SECONDS = 100
    MAX_STEPS = 3000 # 100 seconds * 30 FPS.
    
    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_POINTER = (255, 255, 0)
    COLOR_GOLD = (255, 215, 0)
    COLOR_SILVER = (192, 192, 192)
    COLOR_BRONZE = (205, 127, 50)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (255, 50, 50)
    COLOR_GREEN = (50, 255, 50)
    COLOR_UI_TEXT = (230, 230, 230)
    RAINBOW_COLORS = [(255,0,0), (255,127,0), (255,255,0), (0,255,0), (0,0,255), (75,0,130), (148,0,211)]

    # Wheel properties
    WHEEL_CENTER = (WIDTH // 2, HEIGHT // 2 + 50)
    WHEEL_RADIUS = 150
    SEGMENT_COUNT = 12
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.FONT_SCORE = pygame.font.SysFont("Consolas", 24, bold=True)
        self.FONT_COMBO = pygame.font.SysFont("Impact", 36)
        self.FONT_END = pygame.font.SysFont("Impact", 80)
        self.FONT_FLOATING = pygame.font.SysFont("Arial", 20, bold=True)

        self.wheel_segments = []
        self.last_space_held = False
        self.particles = []
        self.floating_texts = []
        
        # self.reset() is called by the wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TOTAL_TIME_SECONDS * self.FPS
        
        self.wheel_angle = self.np_random.uniform(0, 360)
        self.wheel_velocity = 0.0
        self.wheel_state = 'rotating'  # 'rotating', 'stopping', 'stopped'
        self.stop_timer = 0
        
        self.chain_multiplier = 1
        self.chain_spins_left = 0
        self.consecutive_golds = 0
        
        self.last_space_held = False
        self.particles = []
        self.floating_texts = []
        
        self._setup_wheel_segments()
        
        return self._get_observation(), self._get_info()

    def _setup_wheel_segments(self):
        self.wheel_segments = []
        segments_def = (
            [{"value": 10, "color": self.COLOR_GOLD, "type": "gold", "reward": 1.0}] * 3 +
            [{"value": 5, "color": self.COLOR_SILVER, "type": "silver", "reward": 0.5}] * 4 +
            [{"value": 1, "color": self.COLOR_BRONZE, "type": "bronze", "reward": 0.1}] * 5
        )
        shuffled_segments = list(segments_def)
        self.np_random.shuffle(shuffled_segments)
        self.wheel_segments = shuffled_segments

    def step(self, action):
        movement, space_held_int, shift_held_int = action
        space_held = space_held_int == 1
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held

        reward = 0.0
        
        if not self.game_over:
            self.steps += 1
            self.timer -= 1
            
            self._update_wheel(space_pressed)
            spin_reward = self._process_spin_result()
            reward += spin_reward
            
            self._update_effects()

        terminated = self.score >= self.WIN_SCORE or self.timer <= 0 or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100.0
                self.floating_texts.append(FloatingText(self.WIDTH / 2, self.HEIGHT / 2, "YOU WIN!", self.FONT_END, self.COLOR_GOLD, self.FPS * 3))
            elif self.timer <= 0:
                reward -= 100.0
                self.floating_texts.append(FloatingText(self.WIDTH / 2, self.HEIGHT / 2, "TIME'S UP!", self.FONT_END, self.COLOR_RED, self.FPS * 3))
            self.game_over = True

        truncated = False
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_wheel(self, space_pressed):
        if self.wheel_state == 'rotating':
            # Sinusoidal acceleration cycle over 5 seconds (150 frames)
            cycle_progress = (self.steps % (5 * self.FPS)) / (5 * self.FPS)
            base_speed = 2.0  # degrees per frame
            amp_speed = 1.5   # degrees per frame
            self.wheel_velocity = base_speed + amp_speed * math.sin(cycle_progress * 2 * math.pi)
            self.wheel_angle = (self.wheel_angle + self.wheel_velocity) % 360
            
            if space_pressed:
                # sfx: SpinInitiate
                self.wheel_state = 'stopping'
                # Decelerate over ~0.5 seconds
                self.deceleration = self.wheel_velocity / (self.FPS * 0.5)

        elif self.wheel_state == 'stopping':
            self.wheel_velocity = max(0, self.wheel_velocity - self.deceleration)
            self.wheel_angle = (self.wheel_angle + self.wheel_velocity) % 360
            if self.wheel_velocity == 0:
                self.wheel_state = 'stopped'
                self.stop_timer = self.FPS * 1.5 # Pause for 1.5 seconds

        elif self.wheel_state == 'stopped':
            self.stop_timer -= 1
            if self.stop_timer <= 0:
                self.wheel_state = 'rotating'

    def _process_spin_result(self):
        # This function only does something on the exact frame the wheel stops
        if self.wheel_state != 'stopped' or self.stop_timer != (self.FPS * 1.5 - 1):
            return 0.0

        segment_angle = 360 / self.SEGMENT_COUNT
        # Pointer is at 270 degrees (top of circle)
        pointer_angle = 270
        current_segment_index = int(((self.wheel_angle - pointer_angle + 360) % 360) / segment_angle)
        segment = self.wheel_segments[current_segment_index]

        # Handle chain multiplier
        if self.chain_spins_left > 0:
            self.chain_spins_left -= 1
            if self.chain_spins_left == 0:
                self.chain_multiplier = 1

        # Calculate score and reward
        points_earned = segment['value'] * self.chain_multiplier
        self.score += points_earned
        reward = segment['reward']
        
        # sfx based on segment type
        if segment['type'] == 'gold': # sfx: HitGold
            self._create_particles(self.WHEEL_CENTER[0], 50, 100, [self.COLOR_GOLD, (255,255,100)], 5, 2, self.FPS)
        elif segment['type'] == 'silver': # sfx: HitSilver
            self._create_particles(self.WHEEL_CENTER[0], 50, 60, [self.COLOR_SILVER, self.COLOR_WHITE], 4, 1.5, self.FPS * 0.8)
        else: # sfx: HitBronze
            self._create_particles(self.WHEEL_CENTER[0], 50, 30, [self.COLOR_BRONZE, (139,69,19)], 3, 1, self.FPS * 0.6)

        # Update chain reaction logic
        if self.chain_multiplier == 1: # Can only build combo if not already in one
            if segment['type'] == 'gold':
                self.consecutive_golds += 1
            else:
                self.consecutive_golds = 0
            
            if self.consecutive_golds >= 3:
                # sfx: ChainReactionTrigger
                self.chain_multiplier = 2
                self.chain_spins_left = 3
                self.consecutive_golds = 0
                reward += 5.0
                # Big visual feedback for chain reaction
                self.floating_texts.append(FloatingText(self.WHEEL_CENTER[0], self.WHEEL_CENTER[1], "COMBO!", self.FONT_COMBO, self.COLOR_GOLD, self.FPS * 2))
        
        # Floating text for points
        text = f"+{points_earned}"
        if self.chain_multiplier > 1:
            text += " x2!"
        self.floating_texts.append(FloatingText(self.WHEEL_CENTER[0], 50, text, self.FONT_FLOATING, segment['color'], self.FPS))
        
        return reward

    def _update_effects(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]
        
        for t in self.floating_texts:
            t.update()
        self.floating_texts = [t for t in self.floating_texts if t.lifespan > 0]

    def _create_particles(self, x, y, count, colors, speed, size, lifespan):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            s = self.np_random.uniform(speed * 0.5, speed)
            vx = math.cos(angle) * s
            vy = math.sin(angle) * s
            c = colors[self.np_random.integers(len(colors))]
            p_size = self.np_random.uniform(size * 0.5, size)
            p_lifespan = self.np_random.uniform(lifespan * 0.7, lifespan)
            self.particles.append(Particle(x, y, vx, vy, p_size, c, p_lifespan))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background gradient
        for i in range(self.HEIGHT // 2):
            alpha = 1 - (i / (self.HEIGHT // 2))
            color = (
                int(self.COLOR_BG[0] * alpha + 40 * (1 - alpha)),
                int(self.COLOR_BG[1] * alpha + 30 * (1 - alpha)),
                int(self.COLOR_BG[2] * alpha + 70 * (1 - alpha))
            )
            pygame.draw.rect(self.screen, color, (0, i, self.WIDTH, 1))
            pygame.draw.rect(self.screen, color, (0, self.HEIGHT - i - 1, self.WIDTH, 1))

        # Draw wheel segments
        segment_angle = 360 / self.SEGMENT_COUNT
        for i, segment in enumerate(self.wheel_segments):
            start_angle = self.wheel_angle + i * segment_angle
            end_angle = start_angle + segment_angle
            
            points = [self.WHEEL_CENTER]
            for angle_deg in range(int(start_angle), int(end_angle) + 1):
                angle_rad = math.radians(angle_deg)
                x = self.WHEEL_CENTER[0] + self.WHEEL_RADIUS * math.cos(angle_rad)
                y = self.WHEEL_CENTER[1] - self.WHEEL_RADIUS * math.sin(angle_rad)
                points.append((x, y))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, segment['color'])
                pygame.gfxdraw.filled_polygon(self.screen, points, segment['color'])

        # Draw wheel outline and hub
        pygame.gfxdraw.aacircle(self.screen, self.WHEEL_CENTER[0], self.WHEEL_CENTER[1], self.WHEEL_RADIUS, self.COLOR_WHITE)
        pygame.gfxdraw.aacircle(self.screen, self.WHEEL_CENTER[0], self.WHEEL_CENTER[1], self.WHEEL_RADIUS-1, self.COLOR_WHITE)
        pygame.gfxdraw.filled_circle(self.screen, self.WHEEL_CENTER[0], self.WHEEL_CENTER[1], 20, self.COLOR_BG)
        pygame.gfxdraw.aacircle(self.screen, self.WHEEL_CENTER[0], self.WHEEL_CENTER[1], 20, self.COLOR_WHITE)

        # Draw pointer
        pointer_points = [(self.WHEEL_CENTER[0] - 15, self.WHEEL_CENTER[1] - self.WHEEL_RADIUS - 5),
                          (self.WHEEL_CENTER[0] + 15, self.WHEEL_CENTER[1] - self.WHEEL_RADIUS - 5),
                          (self.WHEEL_CENTER[0], self.WHEEL_CENTER[1] - self.WHEEL_RADIUS - 25)]
        pygame.gfxdraw.aapolygon(self.screen, pointer_points, self.COLOR_POINTER)
        pygame.gfxdraw.filled_polygon(self.screen, pointer_points, self.COLOR_POINTER)

        # Draw effects
        for p in self.particles:
            p.draw(self.screen)
        for t in self.floating_texts:
            t.draw(self.screen)
    
    def _render_ui(self):
        # Score
        score_text = self.FONT_SCORE.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer bar
        timer_ratio = max(0, self.timer / (self.TOTAL_TIME_SECONDS * self.FPS))
        bar_width = 150
        bar_height = 20
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        
        color = (
            int(self.COLOR_RED[0] * (1 - timer_ratio) + self.COLOR_GREEN[0] * timer_ratio),
            int(self.COLOR_RED[1] * (1 - timer_ratio) + self.COLOR_GREEN[1] * timer_ratio),
            int(self.COLOR_RED[2] * (1 - timer_ratio) + self.COLOR_GREEN[2] * timer_ratio)
        )
        
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, int(bar_width * timer_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Combo multiplier display
        if self.chain_spins_left > 0:
            combo_text = self.FONT_COMBO.render(f"x{self.chain_multiplier} COMBO!", True, self.RAINBOW_COLORS[self.steps % len(self.RAINBOW_COLORS)])
            text_rect = combo_text.get_rect(center=(self.WHEEL_CENTER[0], self.WHEEL_CENTER[1]))
            self.screen.blit(combo_text, text_rect)
        elif self.consecutive_golds > 0:
            # Show progress towards combo
            gold_text = self.FONT_COMBO.render("".join(["★"] * self.consecutive_golds), True, self.COLOR_GOLD)
            text_rect = gold_text.get_rect(center=(self.WHEEL_CENTER[0], self.WHEEL_CENTER[1]))
            self.screen.blit(gold_text, text_rect)
            
        # Game Over overlay
        if self.game_over and self.stop_timer <= 0:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME'S UP"
            color = self.COLOR_GOLD if self.score >= self.WIN_SCORE else self.COLOR_RED
            end_text = self.FONT_END.render(end_msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 40))
            self.screen.blit(end_text, text_rect)

            final_score_text = self.FONT_SCORE.render(f"FINAL SCORE: {self.score}", True, self.COLOR_UI_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 40))
            self.screen.blit(final_score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer / self.FPS,
            "chain_multiplier": self.chain_multiplier,
            "chain_spins_left": self.chain_spins_left
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Spin Cycle")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        space_held = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        # Action: [no-op movement, space, no-op shift]
        action = [0, space_held, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose the observation back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait for 'r' to reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
                clock.tick(GameEnv.FPS)

        clock.tick(GameEnv.FPS)
        
    env.close()