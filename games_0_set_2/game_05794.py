
# Generated: 2025-08-28T06:07:18.765975
# Source Brief: brief_05794.md
# Brief Index: 5794

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a fast-paced arcade "monster clicker" game.
    The player controls a cursor to shoot down waves of descending monsters.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press Space to fire at the cursor's location."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of colorful monsters advancing down the screen by targeting them with your cursor and firing before they reach the bottom."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    CURSOR_SPEED = 15

    # --- Colors ---
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (2, 5, 15)
    COLOR_TEXT = (255, 255, 255)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CURSOR_OUTLINE = (0, 0, 0)
    EXPLOSION_COLORS = [(255, 204, 0), (255, 165, 0), (255, 69, 0)]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Pre-render background for performance
        self.background = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self._create_gradient_background()
        
        # Initialize state variables
        self.cursor_pos = None
        self.monsters = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.space_was_pressed = False
        
        self.reset()
    
    def _create_gradient_background(self):
        """Renders a vertical gradient to a surface for blitting."""
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = tuple(
                int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp)
                for i in range(3)
            )
            pygame.draw.line(self.background, color, (0, y), (self.SCREEN_WIDTH, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.monsters = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.space_was_pressed = False
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def _spawn_wave(self):
        """Populates the monster list for a new wave."""
        self.monsters.clear()
        num_monsters = 5 + self.wave * 2
        monster_base_speed = 1.0 + (self.wave - 1) * 0.2
        
        for _ in range(num_monsters):
            size = self.np_random.integers(10, 20)
            monster = {
                "pos": np.array([
                    self.np_random.uniform(size, self.SCREEN_WIDTH - size),
                    self.np_random.uniform(-150, -size)
                ], dtype=np.float32),
                "size": size,
                "speed": monster_base_speed * self.np_random.uniform(0.8, 1.2),
                "color": tuple(self.np_random.integers(100, 256, size=3)),
                "shape": self.np_random.choice(['circle', 'square'])
            }
            self.monsters.append(monster)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.01  # Small survival reward per step
        terminated = False

        movement, space_action, _ = action
        space_pressed = space_action == 1

        # --- Handle Input ---
        self._handle_movement(movement)
        
        # Fire on key press, not hold, to simulate a "click"
        fire_event = space_pressed and not self.space_was_pressed
        if fire_event:
            # // SFX: Laser fire
            if self._handle_fire():
                reward += 1.0  # Reward for hitting a monster
                self.score += 100
        self.space_was_pressed = space_pressed

        # --- Update Game State ---
        self._update_monsters()
        self._update_particles()
        
        # --- Check Termination Conditions ---
        if any(m["pos"][1] > self.SCREEN_HEIGHT + m["size"] for m in self.monsters):
            # // SFX: Failure sound
            terminated = True
            reward = -10.0  # Penalty for losing
            self.game_over = True
        
        if not terminated:
            if not self.monsters:
                # // SFX: Wave clear success
                reward += 10.0  # Bonus for clearing the wave
                self.wave += 1
                self._spawn_wave()

            if self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        """Updates cursor position based on movement action."""
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED  # Up
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED  # Down
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED  # Left
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

    def _handle_fire(self):
        """Checks for monster hits at cursor, removes one if found."""
        for i in range(len(self.monsters) - 1, -1, -1):
            monster = self.monsters[i]
            dist = np.linalg.norm(self.cursor_pos - monster["pos"])
            if dist < monster["size"]:
                # // SFX: Explosion
                self._create_explosion(monster["pos"])
                self.monsters.pop(i)
                return True  # Hit a monster
        return False  # Missed

    def _update_monsters(self):
        for monster in self.monsters:
            monster["pos"][1] += monster["speed"]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] -= 0.2
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['radius'] > 0]

    def _create_explosion(self, pos):
        """Spawns explosion particles at a given position."""
        num_particles = self.np_random.integers(15, 25)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': self.np_random.choice(self.EXPLOSION_COLORS),
                'lifetime': self.np_random.integers(15, 30),
                'radius': self.np_random.integers(4, 8)
            })

    def _get_observation(self):
        self.screen.blit(self.background, (0, 0))
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all dynamic game elements."""
        for monster in self.monsters:
            pos = (int(monster["pos"][0]), int(monster["pos"][1]))
            size = int(monster["size"])
            outline_color = tuple(max(0, c - 80) for c in monster["color"])
            
            if monster["shape"] == 'circle':
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, monster["color"])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, monster["color"])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size - 1, outline_color)
            else:
                rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
                pygame.draw.rect(self.screen, monster["color"], rect)
                pygame.draw.rect(self.screen, outline_color, rect, 2)

        for p in self.particles:
            if p['radius'] > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

        cursor_x, cursor_y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.draw.circle(self.screen, self.COLOR_CURSOR_OUTLINE, (cursor_x, cursor_y), 12, 3)
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y), 10, 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x - 15, cursor_y), (cursor_x - 5, cursor_y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x + 5, cursor_y), (cursor_x + 15, cursor_y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y - 15), (cursor_x, cursor_y - 5), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y + 5), (cursor_x, cursor_y + 15), 2)

    def _render_ui(self):
        """Renders score, wave, and game over text."""
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        wave_text = self.font_large.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        text_rect = wave_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(wave_text, text_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render("GAME OVER", True, (255, 50, 50))
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
        }

    def close(self):
        pygame.quit()

# This block allows for direct execution and playing of the game
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Monster Clicker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        movement_action = 0  # No-op
        space_action = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        if keys[pygame.K_SPACE]: space_action = 1

        action = [movement_action, space_action, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Wave: {info['wave']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000)

        clock.tick(30)

    env.close()