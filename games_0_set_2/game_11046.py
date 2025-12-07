import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:30:52.847622
# Source Brief: brief_01046.md
# Brief Index: 1046
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects
class Crate:
    """Represents a single crate on a conveyor belt."""
    def __init__(self, belt_idx: int, y_pos: float, size: int = 30):
        self.belt_idx = belt_idx
        self.pos_x = 10.0
        self.y_pos = y_pos
        self.size = size
        self.is_scoring = False

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, x: float, y: float, color: tuple):
        self.pos = [x, y]
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.life = random.randint(20, 40)
        self.initial_life = self.life
        self.color = color
        self.radius = random.uniform(2, 5)

    def update(self):
        """Updates particle position and lifetime."""
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += 0.05  # A little gravity
        self.life -= 1

    def draw(self, surface: pygame.Surface):
        """Draws the particle on the given surface."""
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            # Use gfxdraw for anti-aliased circles for a smoother look
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), int(self.radius), self.color + (alpha,))
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), int(self.radius), self.color + (alpha,))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage two conveyor belts to process crates. Adjust belt speeds to match the target "
        "and place new crates to maximize your score before time runs out."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to adjust Belt 1 speed and ←→ for Belt 2. "
        "Press space to place a crate on Belt 1 and shift for Belt 2."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 6000
    TARGET_SCORE = 1000
    TARGET_SPEED = 5.0
    SPEED_TOLERANCE = 0.5
    MIN_SPEED = 0.0
    MAX_SPEED = 10.0
    SPEED_ADJUST_RATE = 0.1
    CRATE_PLACEMENT_COOLDOWN = 15  # steps

    # --- COLORS ---
    COLOR_BG = (25, 35, 45)
    COLOR_BELT = (60, 65, 75)
    COLOR_BELT_SHADOW = (40, 45, 55)
    COLOR_CRATE = (160, 110, 70)
    COLOR_CRATE_SHADOW = (110, 75, 50)
    COLOR_UI_TEXT = (230, 230, 240)
    COLOR_TARGET_MARKER = (0, 255, 120)
    COLOR_SLIDER_BG = (50, 60, 70)
    COLOR_SLIDER_HANDLE = (200, 200, 210)
    COLOR_SPEED_GOOD = (0, 255, 120)
    COLOR_SPEED_BAD = (255, 100, 100)
    COLOR_PARTICLE_SPAWN = (220, 180, 140)
    COLOR_GLOW_SCORING = (0, 255, 120)

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

        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20)
            self.font_speed = pygame.font.SysFont("Consolas", 16, bold=True)
            self.font_title = pygame.font.SysFont("Consolas", 28, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_speed = pygame.font.SysFont(None, 20)
            self.font_title = pygame.font.SysFont(None, 32)

        # Game state variables are initialized in reset()
        self.belt_speeds = [0.0, 0.0]
        self.crates = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.crate_cooldowns = [0, 0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.belt_speeds = [3.0, 3.0]
        self.crates = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.crate_cooldowns = [0, 0]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- 1. Handle Actions ---
        self._handle_actions(action)

        # --- 2. Update Game State ---
        self._update_game_state()

        # --- 3. Calculate Reward and Score ---
        reward = self._calculate_reward()
        self._update_score()

        # --- 4. Check Termination ---
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held_bool, shift_held_bool = action[0], action[1] == 1, action[2] == 1

        # Adjust belt speeds
        if movement == 1: self.belt_speeds[0] += self.SPEED_ADJUST_RATE  # Up
        elif movement == 2: self.belt_speeds[0] -= self.SPEED_ADJUST_RATE  # Down
        if movement == 4: self.belt_speeds[1] += self.SPEED_ADJUST_RATE  # Right
        elif movement == 3: self.belt_speeds[1] -= self.SPEED_ADJUST_RATE  # Left
        
        self.belt_speeds[0] = np.clip(self.belt_speeds[0], self.MIN_SPEED, self.MAX_SPEED)
        self.belt_speeds[1] = np.clip(self.belt_speeds[1], self.MIN_SPEED, self.MAX_SPEED)

        # Place crates on rising edge of button press
        if space_held_bool and not self.prev_space_held and self.crate_cooldowns[0] <= 0:
            self._place_crate(0)
        if shift_held_bool and not self.prev_shift_held and self.crate_cooldowns[1] <= 0:
            self._place_crate(1)

        self.prev_space_held = space_held_bool
        self.prev_shift_held = shift_held_bool

    def _place_crate(self, belt_idx):
        # sfx: Crate placed sound
        y_pos = 125 if belt_idx == 0 else 275
        self.crates.append(Crate(belt_idx, y_pos))
        self.crate_cooldowns[belt_idx] = self.CRATE_PLACEMENT_COOLDOWN
        for _ in range(15):
            self.particles.append(Particle(25, y_pos, self.COLOR_PARTICLE_SPAWN))

    def _update_game_state(self):
        # Update cooldowns
        self.crate_cooldowns[0] = max(0, self.crate_cooldowns[0] - 1)
        self.crate_cooldowns[1] = max(0, self.crate_cooldowns[1] - 1)

        # Update crates
        for crate in self.crates:
            crate.pos_x += self.belt_speeds[crate.belt_idx]
            speed_diff = abs(self.belt_speeds[crate.belt_idx] - self.TARGET_SPEED)
            crate.is_scoring = speed_diff <= self.SPEED_TOLERANCE
        self.crates = [c for c in self.crates if c.pos_x < self.SCREEN_WIDTH + c.size]

        # Update particles
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.life > 0]

    def _calculate_reward(self):
        reward = 0.0
        for crate in self.crates:
            if crate.is_scoring:
                reward += 0.1
        return reward

    def _update_score(self):
        for crate in self.crates:
            if crate.is_scoring:
                # sfx: Point scored tick
                self.score += 1

    def _check_termination(self):
        win = self.score >= self.TARGET_SCORE
        timeout = self.steps >= self.MAX_STEPS
        
        if win:
            return True, 100.0
        if timeout:
            return True, -100.0
        
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "belt_1_speed": self.belt_speeds[0],
            "belt_2_speed": self.belt_speeds[1],
            "crate_count": len(self.crates),
        }

    def _render_game(self):
        # Render belts
        self._render_belt(100, self.belt_speeds[0])
        self._render_belt(250, self.belt_speeds[1])

        # Render particles
        for p in self.particles:
            p.draw(self.screen)

        # Render crates
        for crate in self.crates:
            self._render_crate(crate)

    def _render_belt(self, y_pos, speed):
        belt_rect = pygame.Rect(0, y_pos, self.SCREEN_WIDTH, 50)
        shadow_rect = pygame.Rect(0, y_pos + 50, self.SCREEN_WIDTH, 10)
        pygame.draw.rect(self.screen, self.COLOR_BELT_SHADOW, shadow_rect)
        pygame.draw.rect(self.screen, self.COLOR_BELT, belt_rect)

        # Animate belt texture
        line_spacing = 40
        offset = (self.steps * speed) % line_spacing
        for i in range(-1, self.SCREEN_WIDTH // line_spacing + 2):
            x = i * line_spacing - offset
            pygame.draw.line(self.screen, self.COLOR_BELT_SHADOW, (x, y_pos), (x, y_pos + 50), 2)

    def _render_crate(self, crate: Crate):
        crate_rect = pygame.Rect(crate.pos_x, crate.y_pos, crate.size, crate.size)
        shadow_rect = pygame.Rect(crate.pos_x + 5, crate.y_pos + 5, crate.size, crate.size)
        
        pygame.draw.rect(self.screen, self.COLOR_CRATE_SHADOW, shadow_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_CRATE, crate_rect, border_radius=4)
        
        if crate.is_scoring:
            # sfx: Scoring glow hum
            glow_surf = pygame.Surface((crate.size + 20, crate.size + 20), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_GLOW_SCORING + (50,), (glow_surf.get_width()//2, glow_surf.get_height()//2), crate.size//2 + 10)
            pygame.draw.circle(glow_surf, self.COLOR_GLOW_SCORING + (80,), (glow_surf.get_width()//2, glow_surf.get_height()//2), crate.size//2 + 5)
            self.screen.blit(glow_surf, (crate.pos_x - 10, crate.y_pos - 10), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score and Timer
        score_text = self.font_title.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = max(0, self.MAX_STEPS - self.steps)
        timer_text = self.font_title.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Speed controls
        self._render_speed_control(50, 0, "BELT 1", "UP/DOWN", "SPACE")
        self._render_speed_control(200, 1, "BELT 2", "LEFT/RIGHT", "SHIFT")

        if self.game_over:
            self._render_game_over()
            
    def _render_speed_control(self, y_pos, belt_idx, title, move_keys, place_key):
        # Control Title
        title_text = self.font_ui.render(f"{title} ({move_keys})", True, self.COLOR_UI_TEXT)
        self.screen.blit(title_text, (10, y_pos))
        
        # Place Crate Text
        place_text = self.font_ui.render(f"PLACE CRATE ({place_key})", True, self.COLOR_UI_TEXT)
        self.screen.blit(place_text, (self.SCREEN_WIDTH - place_text.get_width() - 10, y_pos))
        
        # Speed Slider
        slider_y = y_pos + 25
        slider_width = self.SCREEN_WIDTH - 20
        slider_rect = pygame.Rect(10, slider_y, slider_width, 10)
        pygame.draw.rect(self.screen, self.COLOR_SLIDER_BG, slider_rect, border_radius=5)

        # Target Marker
        target_x = 10 + (self.TARGET_SPEED / self.MAX_SPEED) * slider_width
        pygame.draw.rect(self.screen, self.COLOR_TARGET_MARKER, (target_x - 2, slider_y - 5, 4, 20), border_radius=2)
        
        # Current Speed Handle
        current_speed = self.belt_speeds[belt_idx]
        handle_x = 10 + (current_speed / self.MAX_SPEED) * slider_width
        handle_rect = pygame.Rect(0, 0, 10, 24)
        handle_rect.center = (handle_x, slider_y + 5)
        pygame.draw.rect(self.screen, self.COLOR_SLIDER_HANDLE, handle_rect, border_radius=5)

        # Speed Text
        is_good_speed = abs(current_speed - self.TARGET_SPEED) <= self.SPEED_TOLERANCE
        speed_color = self.COLOR_SPEED_GOOD if is_good_speed else self.COLOR_SPEED_BAD
        speed_text_str = f"{current_speed:.2f} / {self.MAX_SPEED:.2f}"
        speed_text = self.font_speed.render(speed_text_str, True, speed_color)
        text_x = np.clip(handle_x, speed_text.get_width() // 2 + 5, self.SCREEN_WIDTH - speed_text.get_width() // 2 - 5)
        self.screen.blit(speed_text, (text_x - speed_text.get_width() // 2, slider_y + 20))


    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        win = self.score >= self.TARGET_SCORE
        end_text_str = "TARGET REACHED!" if win else "TIME UP!"
        end_text_color = self.COLOR_SPEED_GOOD if win else self.COLOR_SPEED_BAD

        end_text = self.font_title.render(end_text_str, True, end_text_color)
        text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
        self.screen.blit(end_text, text_rect)

        final_score_text = self.font_ui.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
        self.screen.blit(final_score_text, score_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_over = False
    
    # Create a display window
    pygame.display.set_caption("Conveyor Belt Manager")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    while running:
        # --- Human Input to Action Mapping ---
        movement = 0  # 0=none
        space_held = 0 # 0=released
        shift_held = 0 # 0=released

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_over:
                    obs, info = env.reset()
                    game_over = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            game_over = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        frame = np.transpose(obs, (1, 0, 2)) # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(60) # Run at 60 FPS for smooth human play

    env.close()