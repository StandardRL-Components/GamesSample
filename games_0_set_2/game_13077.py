import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:42:57.326459
# Source Brief: brief_03077.md
# Brief Index: 3077
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects
class Orb:
    def __init__(self, x, y, radius, color, target_color, speed, score_pos, name):
        self.x = x
        self.y = y
        self.base_radius = radius
        self.radius = radius
        self.color = color
        self.target_color = target_color
        self.speed = speed
        self.score = 0
        self.score_pos = score_pos
        self.name = name
        self.vel_x = 0
        self.vel_y = 0

    def move(self, dx, dy, width, height):
        self.x += dx * self.speed
        self.y += dy * self.speed
        # Clamp to screen boundaries
        self.x = max(self.radius, min(self.x, width - self.radius))
        self.y = max(self.radius, min(self.y, height - self.radius))

    def update_animation(self, steps):
        # Pulsating effect
        self.radius = self.base_radius + math.sin(steps * 0.1) * 2

class Dot:
    def __init__(self, x, y, radius, color, width, height):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.width = width
        self.height = height
        self.anim_offset = random.uniform(0, 2 * math.pi)

    def get_alpha(self, steps):
        # Fade in/out effect
        return 128 + math.sin(steps * 0.05 + self.anim_offset) * 127

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.lifespan = 60 # 2 seconds at 30fps
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        return self.lifespan > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control two orbs, Light and Dark, to collect matching colored dots. "
        "Control automatically switches between orbs periodically."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move the currently controlled orb."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000
    WIN_SCORE = 25
    CONTROL_SWITCH_INTERVAL = 100 # Steps

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_LIGHT_ORB = (255, 255, 255)
    COLOR_DARK_ORB = (40, 40, 40)
    COLOR_RED_DOT = (255, 60, 60)
    COLOR_BLUE_DOT = (60, 150, 255)
    COLOR_LIGHT_TEXT = (240, 240, 240)
    COLOR_DARK_TEXT = (180, 180, 180)
    COLOR_GLOW = (255, 255, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game state variables
        self.steps = 0
        self.light_orb = None
        self.dark_orb = None
        self.controlled_orb = None
        self.ai_orb = None
        self.red_dots = []
        self.blue_dots = []
        self.particles = []
        self.control_switch_timer = 0
        
        # self.reset() # Removed to align with Gymnasium API, reset is called by the user.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        
        self.light_orb = Orb(100, self.HEIGHT / 2, 20, self.COLOR_LIGHT_ORB, self.COLOR_RED_DOT, 6, (20, 10), "Light")
        self.dark_orb = Orb(self.WIDTH - 100, self.HEIGHT / 2, 20, self.COLOR_DARK_ORB, self.COLOR_BLUE_DOT, 6, (self.WIDTH - 120, 10), "Dark")
        
        self.controlled_orb = self.light_orb
        self.ai_orb = self.dark_orb
        
        self.red_dots = [self._spawn_dot(self.COLOR_RED_DOT) for _ in range(10)]
        self.blue_dots = [self._spawn_dot(self.COLOR_BLUE_DOT) for _ in range(10)]
        
        self.particles = []
        self.control_switch_timer = self.CONTROL_SWITCH_INTERVAL

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0

        # --- UPDATE CONTROL ---
        self.control_switch_timer -= 1
        if self.control_switch_timer <= 0:
            self.control_switch_timer = self.CONTROL_SWITCH_INTERVAL
            if self.controlled_orb == self.light_orb:
                self.controlled_orb = self.dark_orb
                self.ai_orb = self.light_orb
            else:
                self.controlled_orb = self.light_orb
                self.ai_orb = self.dark_orb
            # Sound: `control_switch.wav`

        # --- HANDLE INPUT ---
        movement = action[0]
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        # --- CONTINUOUS REWARD ---
        target_dots = self.red_dots if self.controlled_orb.target_color == self.COLOR_RED_DOT else self.blue_dots
        old_dist = self._get_closest_dot_dist(self.controlled_orb, target_dots)
        
        self.controlled_orb.move(dx, dy, self.WIDTH, self.HEIGHT)
        
        new_dist = self._get_closest_dot_dist(self.controlled_orb, target_dots)
        reward += (old_dist - new_dist) * 0.01 # Scaled down to be small
        
        # --- AI MOVEMENT ---
        self._update_ai_orb()

        # --- UPDATE GAME LOGIC ---
        # Orb-Orb collision
        if self._check_collision(self.light_orb, self.dark_orb):
            reward -= 1.0
            self.light_orb.score = max(0, self.light_orb.score - 1)
            self.dark_orb.score = max(0, self.dark_orb.score - 1)
            # Sound: `collision.wav`
            # Knockback
            self._apply_knockback(self.light_orb, self.dark_orb)

        # Orb-Dot collision
        reward += self._check_dot_collisions(self.light_orb)
        reward += self._check_dot_collisions(self.dark_orb)

        # Update animations
        self.light_orb.update_animation(self.steps)
        self.dark_orb.update_animation(self.steps)
        self.particles = [p for p in self.particles if p.update()]

        # --- CHECK TERMINATION ---
        terminated = (self.light_orb.score >= self.WIN_SCORE or
                      self.dark_orb.score >= self.WIN_SCORE or
                      self.steps >= self.MAX_STEPS)
        truncated = False # Per Gymnasium API

        if terminated:
            if self.light_orb.score > self.dark_orb.score:
                reward += 100
            elif self.dark_orb.score > self.light_orb.score:
                reward -= 100
            # Sound: `game_over.wav`

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "light_score": self.light_orb.score,
            "dark_score": self.dark_orb.score,
            "steps": self.steps,
            "control_timer": self.control_switch_timer,
        }

    # --- HELPER METHODS ---

    def _spawn_dot(self, color):
        return Dot(
            random.randint(20, self.WIDTH - 20),
            random.randint(50, self.HEIGHT - 20),
            7, color, self.WIDTH, self.HEIGHT
        )

    def _get_closest_dot_dist(self, orb, dots):
        if not dots:
            return self.WIDTH # Return a large number if no dots exist
        min_dist_sq = float('inf')
        for dot in dots:
            dist_sq = (orb.x - dot.x)**2 + (orb.y - dot.y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        return math.sqrt(min_dist_sq)

    def _update_ai_orb(self):
        target_dots = self.red_dots if self.ai_orb.target_color == self.COLOR_RED_DOT else self.blue_dots
        if not target_dots: return

        closest_dot = min(target_dots, key=lambda d: (self.ai_orb.x - d.x)**2 + (self.ai_orb.y - d.y)**2)
        
        dx, dy = 0, 0
        if closest_dot.x > self.ai_orb.x: dx = 1
        elif closest_dot.x < self.ai_orb.x: dx = -1
        if closest_dot.y > self.ai_orb.y: dy = 1
        elif closest_dot.y < self.ai_orb.y: dy = -1
        
        self.ai_orb.move(dx, dy, self.WIDTH, self.HEIGHT)

    def _check_collision(self, obj1, obj2):
        dist_sq = (obj1.x - obj2.x)**2 + (obj1.y - obj2.y)**2
        return dist_sq < (obj1.radius + obj2.radius)**2
    
    def _apply_knockback(self, orb1, orb2):
        dx, dy = orb1.x - orb2.x, orb1.y - orb2.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist == 0: dist = 1
        
        overlap = (orb1.radius + orb2.radius) - dist
        
        orb1.x += (dx / dist) * overlap / 2
        orb1.y += (dy / dist) * overlap / 2
        orb2.x -= (dx / dist) * overlap / 2
        orb2.y -= (dy / dist) * overlap / 2

        orb1.move(0, 0, self.WIDTH, self.HEIGHT) # Clamp
        orb2.move(0, 0, self.WIDTH, self.HEIGHT) # Clamp


    def _check_dot_collisions(self, orb):
        reward_from_dots = 0
        target_dots = self.red_dots if orb.target_color == self.COLOR_RED_DOT else self.blue_dots
        
        dots_to_remove = []
        for dot in target_dots:
            if self._check_collision(orb, dot):
                dots_to_remove.append(dot)
                orb.score += 1
                reward_from_dots += 1.0
                # Sound: `collect_dot.wav`
                for _ in range(15):
                    self.particles.append(Particle(dot.x, dot.y, dot.color))

        if dots_to_remove:
            if orb.target_color == self.COLOR_RED_DOT:
                self.red_dots = [d for d in self.red_dots if d not in dots_to_remove]
                for _ in dots_to_remove:
                    self.red_dots.append(self._spawn_dot(self.COLOR_RED_DOT))
            else:
                self.blue_dots = [d for d in self.blue_dots if d not in dots_to_remove]
                for _ in dots_to_remove:
                    self.blue_dots.append(self._spawn_dot(self.COLOR_BLUE_DOT))
        
        return reward_from_dots

    # --- RENDER METHODS ---

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p.lifespan / 60))))
            color = p.color + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p.x - 2), int(p.y - 2)))

        # Dots
        for dot in self.red_dots + self.blue_dots:
            alpha = dot.get_alpha(self.steps)
            color = dot.color
            pygame.gfxdraw.aacircle(self.screen, int(dot.x), int(dot.y), int(dot.radius), (*color, int(alpha)))
            pygame.gfxdraw.filled_circle(self.screen, int(dot.x), int(dot.y), int(dot.radius), (*color, int(alpha)))

        # Orbs
        self._draw_orb(self.light_orb)
        self._draw_orb(self.dark_orb)

        # Controlled orb indicator
        if self.controlled_orb:
            self._draw_glow(self.controlled_orb)
    
    def _draw_orb(self, orb):
        # Draw filled circle
        pygame.gfxdraw.filled_circle(self.screen, int(orb.x), int(orb.y), int(orb.radius), orb.color)
        # Draw anti-aliased outline
        pygame.gfxdraw.aacircle(self.screen, int(orb.x), int(orb.y), int(orb.radius), (200, 200, 200))

    def _draw_glow(self, orb):
        for i in range(4):
            alpha = 60 - i * 15
            radius = orb.radius + i * 2
            color = (*self.COLOR_GLOW, alpha)
            
            # Create a temporary surface for transparency
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surf, int(radius), int(radius), int(radius), color)
            self.screen.blit(temp_surf, (int(orb.x - radius), int(orb.y - radius)))

    def _render_ui(self):
        # Light Orb Score
        score_text = self.font_large.render(f"{self.light_orb.score}", True, self.COLOR_LIGHT_TEXT)
        self.screen.blit(score_text, self.light_orb.score_pos)

        # Dark Orb Score
        score_text = self.font_large.render(f"{self.dark_orb.score}", True, self.COLOR_DARK_TEXT)
        text_rect = score_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(score_text, text_rect)
        
        # Control switch timer bar
        bar_width = self.WIDTH / 3
        bar_height = 8
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = 15
        
        progress = self.control_switch_timer / self.CONTROL_SWITCH_INTERVAL
        
        # Background of the bar
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        
        # Filled part of the bar
        fill_color = self.light_orb.color if self.controlled_orb == self.light_orb else self.COLOR_DARK_TEXT
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, bar_width * progress, bar_height), border_radius=4)
        
        # Timer text
        timer_text = self.font_small.render(f"Switch in: {self.control_switch_timer / self.FPS:.1f}s", True, self.COLOR_DARK_TEXT)
        timer_rect = timer_text.get_rect(center=(self.WIDTH / 2, bar_y + bar_height + 12))
        self.screen.blit(timer_text, timer_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you'll need to `pip install pygame`
    # It will open a window and let you control the agent.
    
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Polarity Shift")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0

    print("\n--- Manual Control ---")
    print("Arrows: Move")
    print("Q: Quit")

    while not terminated:
        # --- Human Input ---
        movement = 0 # No-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_q]: terminated = True

        action = [movement, 0, 0] # Space and Shift are not used

        # --- Environment Step ---
        obs, reward, term, trunc, info = env.step(action)
        terminated = term or trunc
        total_reward += reward

        # --- Render to screen ---
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"\nGame Over!")
    print(f"Final Info: {info}")
    print(f"Total Reward: {total_reward:.2f}")

    env.close()