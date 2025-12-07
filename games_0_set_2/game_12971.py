import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:44:17.425927
# Source Brief: brief_02971.md
# Brief Index: 2971
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, dx, dy, radius):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        self.dx = dx
        self.dy = dy
        self.radius = radius

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.dy += 0.1  # Gravity
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), color)
            pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), int(self.radius), color)


class OnScreenMessage:
    """A simple class for temporary text on screen."""
    def __init__(self, text, position, color, font, life=60):
        self.text = text
        self.position = position
        self.color = color
        self.font = font
        self.life = life
        self.max_life = life

    def update(self):
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * math.sin(math.pi * (self.life / self.max_life)))
            text_surf = self.font.render(self.text, True, self.color)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=self.position)
            surface.blit(text_surf, text_rect)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack falling blocks as precisely as you can to build a tower. "
        "Earn bonus points for perfect placements and chain combos to achieve a high score before time runs out."
    )
    user_guide = (
        "Use the ← and → arrow keys to move the falling block. Press space to drop it instantly. "
        "Stack blocks perfectly for bonus points."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 3600  # 60 seconds * 60 FPS
    WIN_SCORE = 500

    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_PERFECT = (100, 255, 150)
    COLOR_GOOD = (255, 220, 100)
    COLOR_BAD = (255, 100, 100)

    BLOCK_COLORS = [
        (66, 135, 245),   # Blue
        (245, 66, 66),    # Red
        (66, 245, 135),   # Green
        (245, 227, 66),   # Yellow
        (168, 66, 245),   # Purple
        (245, 135, 66),   # Orange
    ]

    PLAY_AREA_WIDTH = 240
    PLAY_AREA_HEIGHT = 360
    BLOCK_SIZE = 24
    PLAYER_SPEED = 240 # pixels per second
    INITIAL_FALL_SPEED = 48 # pixels per second

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Verdana", 20, bold=True)

        self.play_area_rect = pygame.Rect(
            (self.WIDTH - self.PLAY_AREA_WIDTH) // 2,
            self.HEIGHT - self.PLAY_AREA_HEIGHT - 20,
            self.PLAY_AREA_WIDTH,
            self.PLAY_AREA_HEIGHT
        )

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stacked_blocks = []
        self.falling_block_pos = None
        self.falling_block_color = None
        self.fall_speed = 0
        self.perfect_stack_combo = 0
        self.particles = []
        self.onscreen_messages = []
        self.last_space_held = False

        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stacked_blocks = []
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.perfect_stack_combo = 0
        self.particles = []
        self.onscreen_messages = []
        self.last_space_held = False
        self._spawn_new_block()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        self._update_effects()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle player input ---
        if movement == 3: # Left
            self.falling_block_pos[0] -= self.PLAYER_SPEED / self.FPS
        if movement == 4: # Right
            self.falling_block_pos[0] += self.PLAYER_SPEED / self.FPS

        # Clamp position to play area
        self.falling_block_pos[0] = max(
            self.play_area_rect.left,
            min(self.falling_block_pos[0], self.play_area_rect.right - self.BLOCK_SIZE)
        )
        
        # --- Handle instant drop ---
        # Trigger on press, not hold
        if space_held and not self.last_space_held:
            # // Sound effect: fast whoosh
            self.fall_speed *= 10
        self.last_space_held = space_held

        # --- Update physics ---
        self.falling_block_pos[1] += self.fall_speed / self.FPS

        # --- Check for landing ---
        collision_y = self._check_collision()
        if collision_y is not None:
            self.falling_block_pos[1] = collision_y - self.BLOCK_SIZE
            reward += self._handle_landing()
            self._spawn_new_block()

        # --- Check termination conditions ---
        terminated = False
        truncated = False
        if self.score >= self.WIN_SCORE:
            # // Sound effect: victory fanfare
            reward += 100
            terminated = True
            self.game_over = True
            self.onscreen_messages.append(OnScreenMessage("VICTORY!", (self.WIDTH // 2, self.HEIGHT // 2), self.COLOR_PERFECT, self.font_ui, 180))
        elif self.steps >= self.MAX_STEPS:
            # // Sound effect: failure buzzer
            reward += -100
            terminated = True # This is a time limit, could also be truncation
            self.game_over = True
            self.onscreen_messages.append(OnScreenMessage("TIME UP!", (self.WIDTH // 2, self.HEIGHT // 2), self.COLOR_BAD, self.font_ui, 180))

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_new_block(self):
        x = self.play_area_rect.left + self.np_random.uniform(0, self.play_area_rect.width - self.BLOCK_SIZE)
        y = self.play_area_rect.top
        self.falling_block_pos = [x, y]
        self.falling_block_color = self.BLOCK_COLORS[self.np_random.integers(0, len(self.BLOCK_COLORS))]
        
        # Increase difficulty based on score
        base_speed_multiplier = 1 + (self.score // 200) * 0.25
        self.fall_speed = self.INITIAL_FALL_SPEED * base_speed_multiplier

    def _check_collision(self):
        """Returns the y-coordinate of the surface the block would collide with, or None."""
        block_rect = pygame.Rect(self.falling_block_pos[0], self.falling_block_pos[1], self.BLOCK_SIZE, self.BLOCK_SIZE)

        # Check collision with floor
        if block_rect.bottom >= self.play_area_rect.bottom:
            return self.play_area_rect.bottom

        # Check collision with stacked blocks
        for s_block, _ in self.stacked_blocks:
            if block_rect.colliderect(s_block):
                # Check if we are mostly above the block, not clipping from the side
                if block_rect.bottom > s_block.top and abs(block_rect.centerx - s_block.centerx) < self.BLOCK_SIZE:
                    return s_block.top
        return None
    
    def _handle_landing(self):
        # // Sound effect: soft thud
        reward = 0
        landed_rect = pygame.Rect(self.falling_block_pos[0], self.falling_block_pos[1], self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        # Find block directly underneath
        support_block = None
        min_dist = float('inf')
        for s_block, _ in self.stacked_blocks:
            # Check if landed_rect is on top of s_block
            if abs(landed_rect.bottom - s_block.top) < 1:
                dist_x = abs(landed_rect.centerx - s_block.centerx)
                if dist_x < self.BLOCK_SIZE and dist_x < min_dist:
                    support_block = s_block
                    min_dist = dist_x
        
        # Calculate alignment
        offset = 0
        if support_block:
            offset = abs(landed_rect.centerx - support_block.centerx)
        else: # Landed on floor
            # No penalty or bonus for first block, just place it.
            pass

        if offset <= 2: # Perfect stack
            # // Sound effect: positive chime
            self.perfect_stack_combo += 1
            reward += 5
            self.score += 5
            msg = f"PERFECT! +5"
            if self.perfect_stack_combo > 1:
                chain_bonus = 10
                reward += chain_bonus
                self.score += chain_bonus
                msg = f"CHAIN x{self.perfect_stack_combo}! +{5+chain_bonus}"
            self.onscreen_messages.append(OnScreenMessage(msg, (landed_rect.centerx, landed_rect.top - 20), self.COLOR_PERFECT, self.font_msg))
            self._spawn_particles(landed_rect.center, self.COLOR_PERFECT, 30)

        elif offset <= self.BLOCK_SIZE / 4: # Good stack
            # // Sound effect: neutral click
            self.perfect_stack_combo = 0
            reward += 1
            self.score += 1
            self.onscreen_messages.append(OnScreenMessage("+1", (landed_rect.centerx, landed_rect.top - 20), self.COLOR_GOOD, self.font_msg))
            self._spawn_particles(landed_rect.center, self.COLOR_GOOD, 10)

        else: # Bad stack
            # // Sound effect: dull clank
            self.perfect_stack_combo = 0
            # No direct reward change, but opportunity cost
            self._spawn_particles(landed_rect.center, self.COLOR_BAD, 5, speed_mult=0.5)

        self.stacked_blocks.append((landed_rect, self.falling_block_color))
        return reward

    def _update_effects(self):
        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        # Update messages
        self.onscreen_messages = [m for m in self.onscreen_messages if m.life > 0]
        for m in self.onscreen_messages:
            m.update()

    def _spawn_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            life = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(1, 4)
            self.particles.append(Particle(pos[0], pos[1], color, life, dx, dy, radius))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.play_area_rect.left, self.play_area_rect.right + 1, self.BLOCK_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.play_area_rect.top), (x, self.play_area_rect.bottom))
        for y in range(self.play_area_rect.top, self.play_area_rect.bottom + 1, self.BLOCK_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.play_area_rect.left, y), (self.play_area_rect.right, y))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, self.play_area_rect, 2, border_radius=3)
        
        # Draw stacked blocks
        for block_rect, color in self.stacked_blocks:
            self._draw_block(block_rect, color, 0.7)

        # Draw falling block
        if self.falling_block_pos:
            block_rect = pygame.Rect(int(self.falling_block_pos[0]), int(self.falling_block_pos[1]), self.BLOCK_SIZE, self.BLOCK_SIZE)
            self._draw_block(block_rect, self.falling_block_color, 1.0, has_glow=True)
            
            # Draw landing prediction line
            collision_y = self._check_collision()
            if collision_y is None:
                collision_y = self.play_area_rect.bottom
            
            start_pos = (block_rect.centerx, block_rect.bottom)
            end_pos = (block_rect.centerx, collision_y)
            if end_pos[1] > start_pos[1]:
                pygame.draw.line(self.screen, self.falling_block_color + (100,), start_pos, end_pos, 1)


        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _draw_block(self, rect, color, saturation, has_glow=False):
        # Desaturate for stacked blocks
        h, s, v, a = pygame.Color(*color).hsva
        desaturated_color = pygame.Color(0, 0, 0)
        desaturated_color.hsva = (h, s * saturation, v, a)
        
        border_color = (
            max(0, desaturated_color.r - 40),
            max(0, desaturated_color.g - 40),
            max(0, desaturated_color.b - 40)
        )
        
        if has_glow:
            glow_surf = pygame.Surface((self.BLOCK_SIZE * 2, self.BLOCK_SIZE * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE - 2, color + (40,))
            pygame.gfxdraw.aacircle(glow_surf, self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE - 2, color + (40,))
            self.screen.blit(glow_surf, (rect.x - self.BLOCK_SIZE/2, rect.y - self.BLOCK_SIZE/2))


        pygame.draw.rect(self.screen, desaturated_color, rect, border_radius=3)
        pygame.draw.rect(self.screen, border_color, rect, 1, border_radius=3)

    def _render_ui(self):
        # Draw score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Draw time
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_color = self.COLOR_UI_TEXT if time_left > 10 else self.COLOR_BAD
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

        # Draw combo
        if self.perfect_stack_combo > 1:
            combo_text = self.font_msg.render(f"COMBO x{self.perfect_stack_combo}", True, self.COLOR_PERFECT)
            combo_rect = combo_text.get_rect(midtop=(self.WIDTH // 2, 10))
            self.screen.blit(combo_text, combo_rect)

        # Draw onscreen messages
        for msg in self.onscreen_messages:
            msg.draw(self.screen)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
            "perfect_stack_combo": self.perfect_stack_combo,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Manual play example
    # This block is not used by the evaluation environment, but is useful for testing.
    # It requires a display to be available.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Block Stacker")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0
    
    # --- Action state for human player ---
    movement = 0 # 0=none, 3=left, 4=right
    space_held = 0 # 0=released, 1=held

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    space_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        else:
            movement = 0

        action = [movement, space_held, 0] # Movement, Space, Shift
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        terminated = term
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
    env.close()