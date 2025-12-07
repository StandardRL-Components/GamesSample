import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:08:11.644135
# Source Brief: brief_00823.md
# Brief Index: 823
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for Words
class Word:
    """Represents a single word entity on the screen."""
    def __init__(self, text, pos, font):
        self.text = text
        self.pos = pygame.Vector2(pos)
        self.font = font
        self.state = 'normal'  # 'normal', 'highlighted', 'exploding'
        self.explosion_radius = 0
        self.explosion_duration = 0.4  # seconds
        self.explosion_timer = 0
        self.alpha = 255
        self._recalculate_render()

    def _recalculate_render(self):
        """Pre-renders text surfaces for performance."""
        self.surface = self.font.render(self.text, True, (230, 230, 230))
        self.rect = self.surface.get_rect(center=self.pos)
        self.highlight_surface = self.font.render(self.text, True, (100, 255, 100))

    def update(self, dt):
        """Updates the word's state, particularly for explosions."""
        if self.state == 'exploding':
            self.explosion_timer += dt
            progress = self.explosion_timer / self.explosion_duration
            self.explosion_radius = int(progress * 80) # Max radius of 80
            self.alpha = int(255 * (1 - progress**2))
            if self.explosion_timer >= self.explosion_duration:
                return False  # Signal to remove this word
        return True

    def draw(self, screen):
        """Draws the word based on its current state."""
        if self.state == 'normal':
            screen.blit(self.surface, self.rect)
        elif self.state == 'highlighted':
            screen.blit(self.highlight_surface, self.rect)
            # Add a subtle glow
            glow_rect = self.rect.inflate(10, 10)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (100, 255, 100, 50), glow_surf.get_rect().center, glow_rect.width // 2)
            screen.blit(glow_surf, glow_rect.topleft)
        elif self.state == 'exploding':
            # Draw expanding shockwave
            if self.explosion_radius > 0:
                pygame.gfxdraw.aacircle(screen, int(self.pos.x), int(self.pos.y), self.explosion_radius, (255, 255, 100))
                pygame.gfxdraw.aacircle(screen, int(self.pos.x), int(self.pos.y), max(0, self.explosion_radius - 2), (255, 255, 100))
            
            # Draw fading text
            temp_surf = self.font.render(self.text, True, (255, 255, 100))
            temp_surf.set_alpha(max(0, self.alpha))
            screen.blit(temp_surf, self.rect)

# Helper class for Particles
class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, pos, color):
        self.pos = pygame.Vector2(pos)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(20, 80)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.lifespan = random.uniform(0.5, 1.2)
        self.age = 0
        self.color = color
        self.radius = random.randint(2, 4)

    def update(self, dt):
        """Updates particle position and age."""
        self.pos += self.vel * dt
        self.vel *= 0.95 # friction
        self.age += dt
        return self.age < self.lifespan

    def draw(self, screen):
        """Draws the particle with alpha blending."""
        progress = self.age / self.lifespan
        alpha = int(255 * (1 - progress))
        color_with_alpha = self.color + (alpha,)
        
        surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, color_with_alpha, (self.radius, self.radius), self.radius)
        screen.blit(surf, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Fly your cursor through a field of words, highlighting them and pressing space to trigger chain reactions for points."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Move over a word to highlight it, then press space to destroy it. Hold shift to boost."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 60
    GAME_DURATION = 60.0
    WORDS_TO_WIN = 50

    COLOR_BG_TOP = (10, 10, 30)
    COLOR_BG_BOTTOM = (30, 10, 50)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_BOOST = (255, 100, 255)
    COLOR_UI = (200, 200, 255)
    COLOR_PARTICLE = (255, 255, 150)

    PLAYER_SPEED = 300
    BOOST_SPEED = 600
    PLAYER_RADIUS = 8
    
    INITIAL_SPAWN_INTERVAL = 2.0
    MIN_SPAWN_INTERVAL = 0.5
    SPAWN_RATE_INCREASE_PER_WORD = 0.03

    WORD_LIST = [
        "PYTHON", "AGENT", "REWARD", "ACTION", "STATE", "POLICY", "VECTOR",
        "TENSOR", "MODEL", "LEARN", "ORBIT", "DECAY", "PULSE", "LASER", "BEAM",
        "SPACE", "SHIFT", "GAMMA", "DELTA", "ALPHA", "OMEGA", "NEURAL", "NETWORK",
        "FLUX", "MATRIX", "KERNEL", "PROXY", "SERVER", "CLIENT", "NODE", "DATA"
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.dt = 1 / self.FPS

        self.ui_font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.word_font = pygame.font.SysFont("Segoe UI", 20, bold=True)

        self.player_pos = None
        self.player_trail = None
        self.words = None
        self.particles = None
        self.score = None
        self.time_remaining = None
        self.words_cleared = None
        self.game_over = None
        self.spawn_timer = None
        self.spawn_interval = None
        self.last_space_held = None
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_trail = []
        self.words = []
        self.particles = []
        
        self.score = 0
        self.steps = 0
        self.time_remaining = self.GAME_DURATION
        self.words_cleared = 0
        self.game_over = False
        
        self.spawn_timer = 0
        self.spawn_interval = self.INITIAL_SPAWN_INTERVAL
        self.last_space_held = False

        for _ in range(5):
            self._spawn_word()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        speed = self.BOOST_SPEED if shift_held else self.PLAYER_SPEED
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * speed * self.dt

        self.player_pos.x = np.clip(self.player_pos.x, 0, self.SCREEN_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.SCREEN_HEIGHT)

        self.player_trail.append(self.player_pos.copy())
        if len(self.player_trail) > 15: self.player_trail.pop(0)
        if not shift_held and self.player_trail: self.player_trail.pop(0)

        self.time_remaining -= self.dt

        highlighted_word = None
        for word in self.words:
            if word.state == 'normal' and word.rect.collidepoint(self.player_pos):
                word.state = 'highlighted'
            elif word.state == 'highlighted' and not word.rect.collidepoint(self.player_pos):
                word.state = 'normal'
            if word.state == 'highlighted':
                highlighted_word = word
        
        space_pressed = space_held and not self.last_space_held
        if space_pressed and highlighted_word:
            # SFX: Player_Shoot.wav
            highlighted_word.state = 'exploding'
            reward += 1.0
            self._create_particles(highlighted_word.pos)
            self.words_cleared += 1
            self.score += 10
            self.spawn_interval = max(self.MIN_SPAWN_INTERVAL, self.INITIAL_SPAWN_INTERVAL - self.words_cleared * self.SPAWN_RATE_INCREASE_PER_WORD)
        self.last_space_held = space_held

        newly_exploded_words = []
        self.words = [word for word in self.words if word.update(self.dt)]
        for word in self.words:
            if word.state == 'exploding' and word.explosion_timer < self.dt * 2:
                newly_exploded_words.append(word)
        
        if newly_exploded_words:
            for exploding_word in newly_exploded_words:
                for other_word in self.words:
                    if other_word.state == 'normal' and exploding_word.pos.distance_to(other_word.pos) < exploding_word.explosion_radius:
                        # SFX: Chain_Reaction.wav
                        other_word.state = 'exploding'
                        reward += 0.5
                        self._create_particles(other_word.pos)
                        self.words_cleared += 1
                        self.score += 5

        self.particles = [p for p in self.particles if p.update(self.dt)]

        self.spawn_timer += self.dt
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            self._spawn_word()

        terminated = self.time_remaining <= 0 or self.words_cleared >= self.WORDS_TO_WIN
        if terminated and not self.game_over:
            self.game_over = True
            if self.words_cleared >= self.WORDS_TO_WIN:
                reward += 100.0
                self.score += 1000
            else:
                reward -= 100.0
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_word(self):
        if len(self.words) > 20: return
        text = random.choice(self.WORD_LIST)
        for _ in range(10):
            pos = (random.randint(50, self.SCREEN_WIDTH - 50), random.randint(50, self.SCREEN_HEIGHT - 50))
            if not any(word.pos.distance_to(pos) < 60 for word in self.words):
                self.words.append(Word(text, pos, self.word_font))
                return

    def _create_particles(self, pos, count=20):
        for _ in range(count): self.particles.append(Particle(pos, self.COLOR_PARTICLE))

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        for p in self.particles: p.draw(self.screen)
        for word in self.words: word.draw(self.screen)

        if len(self.player_trail) > 1:
            points = [(int(p.x), int(p.y)) for p in self.player_trail]
            pygame.draw.aalines(self.screen, self.COLOR_PLAYER_BOOST, False, points, 1)

        is_boosting = len(self.player_trail) > 5
        player_color = self.COLOR_PLAYER_BOOST if is_boosting else self.COLOR_PLAYER
        px, py = int(self.player_pos.x), int(self.player_pos.y)
        
        glow_radius = self.PLAYER_RADIUS + 5
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, player_color + (60,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (px - glow_radius, py - glow_radius))

        pygame.draw.circle(self.screen, player_color, (px, py), self.PLAYER_RADIUS, 2)
        pygame.draw.line(self.screen, player_color, (px - self.PLAYER_RADIUS - 3, py), (px + self.PLAYER_RADIUS + 3, py), 2)
        pygame.draw.line(self.screen, player_color, (px, py - self.PLAYER_RADIUS - 3), (px, py + self.PLAYER_RADIUS + 3), 2)
        
    def _render_ui(self):
        score_surf = self.ui_font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 5))
        
        words_surf = self.ui_font.render(f"WORDS: {self.words_cleared}/{self.WORDS_TO_WIN}", True, self.COLOR_UI)
        self.screen.blit(words_surf, (10, 30))

        time_surf = self.ui_font.render(f"TIME: {max(0, self.time_remaining):.1f}", True, self.COLOR_UI)
        self.screen.blit(time_surf, time_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 5)))

        if self.game_over:
            win = self.words_cleared >= self.WORDS_TO_WIN
            result_text = "VICTORY" if win else "TIME UP"
            result_color = (100, 255, 100) if win else (255, 100, 100)
            result_font = pygame.font.SysFont("Consolas", 60, bold=True)
            result_surf = result_font.render(result_text, True, result_color)
            self.screen.blit(result_surf, result_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "words_cleared": self.words_cleared}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(info, dict)
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(reward, float) and isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the game and play it with your keyboard.
    # It is not used by the evaluation environment.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible display driver
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Burst")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0.0

        clock.tick(GameEnv.FPS)
        
    env.close()