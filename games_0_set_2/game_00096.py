
# Generated: 2025-08-27T12:35:09.966748
# Source Brief: brief_00096.md
# Brief Index: 96

        
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
    Fruit Slicer: A fast-paced arcade game where the player slices falling fruit
    while dodging bombs.

    The player controls a vertical slicer that moves horizontally. Pressing the
    action button triggers a slice along the slicer's current column, destroying
    any fruit or bombs in its path.

    **Objective:**
    - Reach a score of 500 to win.
    - Slicing a fruit awards points.
    - Slicing a fruit near a bomb gives bonus points.
    - Slicing 3 bombs ends the game.

    **Difficulty:**
    - The rate of falling objects and their speed increases over time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to move the slicer. Press space to slice."
    )

    # Short, user-facing description of the game
    game_description = (
        "Slice falling fruit and dodge bombs to score points. Slice near bombs for a bonus!"
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 10000
    WIN_SCORE = 500
    MAX_BOMB_HITS = 3

    # Colors
    COLOR_BG_TOP = (15, 25, 40)
    COLOR_BG_BOTTOM = (30, 50, 70)
    COLOR_SLICER = (220, 220, 255, 150)
    COLOR_SLICER_ACTIVE = (255, 255, 255)
    COLOR_BOMB = (40, 40, 40)
    COLOR_BOMB_FUSE = (255, 80, 80)
    COLOR_APPLE = (220, 50, 50)
    COLOR_APPLE_SHINE = (250, 150, 150)
    COLOR_ORANGE = (255, 165, 0)
    COLOR_ORANGE_SHINE = (255, 215, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_BOMB_ICON = (200, 0, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.bomb_hits = 0
        self.game_over = False
        self.slicer_x = 0
        self.slicer_flash_timer = 0
        self.prev_space_held = False
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.explosions = []
        self.fruit_spawn_timer = 0
        self.bomb_spawn_timer = 0
        self.base_fruit_spawn_interval = 1.0  # seconds
        self.base_bomb_spawn_interval = 2.0   # seconds
        self.fall_speed_multiplier = 1.0

        # Run validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.bomb_hits = 0
        self.game_over = False

        self.slicer_x = self.WIDTH // 2
        self.slicer_flash_timer = 0
        self.prev_space_held = False

        self.fruits = []
        self.bombs = []
        self.particles = []
        self.explosions = []

        self.fruit_spawn_timer = self.base_fruit_spawn_interval * self.FPS
        self.bomb_spawn_timer = self.base_bomb_spawn_interval * self.FPS
        self.fall_speed_multiplier = 1.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        # shift_held is unused in this game

        reward = -0.01  # Small penalty per step to encourage action
        self.steps += 1

        # --- 1. Handle Input & Slicer State ---
        self._handle_input(movement)
        is_slicing = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        if is_slicing:
            self.slicer_flash_timer = 3  # Flash for 3 frames
            # SFX: whoosh.wav

        # --- 2. Update Game Logic ---
        reward += self._update_objects(is_slicing)
        self._spawn_objects()
        self._update_difficulty()
        self._update_effects()

        # --- 3. Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win bonus
            elif self.bomb_hits >= self.MAX_BOMB_HITS:
                # The bomb hit already gave a -50 penalty
                pass

        # --- 4. Return Observation and Info ---
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _handle_input(self, movement):
        slicer_speed = 15
        if movement == 3:  # Left
            self.slicer_x -= slicer_speed
        elif movement == 4:  # Right
            self.slicer_x += slicer_speed
        self.slicer_x = np.clip(self.slicer_x, 0, self.WIDTH)

    def _update_objects(self, is_slicing):
        step_reward = 0

        # Update and slice fruits
        sliced_fruits = []
        for fruit in self.fruits:
            fruit["pos"][1] += fruit["speed"] * self.fall_speed_multiplier
            if is_slicing and abs(fruit["pos"][0] - self.slicer_x) < fruit["radius"]:
                sliced_fruits.append(fruit)
        
        for fruit in sliced_fruits:
            # SFX: slice.wav
            self.fruits.remove(fruit)
            self._create_particles(fruit["pos"], fruit["color"])
            step_reward += 1

            # Check for risky bonus
            is_risky = False
            for bomb in self.bombs:
                dist = math.hypot(fruit["pos"][0] - bomb["pos"][0], fruit["pos"][1] - bomb["pos"][1])
                if dist < (fruit["radius"] + bomb["radius"] + 20):
                    is_risky = True
                    break
            if is_risky:
                step_reward += 5
                # SFX: bonus_point.wav

        # Update and slice bombs
        sliced_bombs = []
        for bomb in self.bombs:
            bomb["pos"][1] += bomb["speed"] * self.fall_speed_multiplier
            if is_slicing and abs(bomb["pos"][0] - self.slicer_x) < bomb["radius"]:
                sliced_bombs.append(bomb)
        
        for bomb in sliced_bombs:
            # SFX: explosion.wav
            self.bombs.remove(bomb)
            self.bomb_hits += 1
            step_reward -= 50
            self._create_explosion(bomb["pos"])

        # Remove off-screen objects
        self.fruits = [f for f in self.fruits if f["pos"][1] < self.HEIGHT + f["radius"]]
        self.bombs = [b for b in self.bombs if b["pos"][1] < self.HEIGHT + b["radius"]]
        
        self.score += step_reward # Update score based on reward for slicing
        return step_reward

    def _spawn_objects(self):
        # Spawn fruits
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            self._spawn_fruit()
            spawn_interval = self.base_fruit_spawn_interval / max(1, self.fall_speed_multiplier * 0.8)
            self.fruit_spawn_timer = int(self.np_random.uniform(0.8, 1.2) * spawn_interval * self.FPS)

        # Spawn bombs
        self.bomb_spawn_timer -= 1
        if self.bomb_spawn_timer <= 0:
            self._spawn_bomb()
            spawn_interval = self.base_bomb_spawn_interval / max(1, self.fall_speed_multiplier * 0.5)
            self.bomb_spawn_timer = int(self.np_random.uniform(0.9, 1.1) * spawn_interval * self.FPS)

    def _spawn_fruit(self):
        fruit_type = self.np_random.choice(["apple", "orange"])
        radius = 20
        pos = [self.np_random.integers(radius, self.WIDTH - radius), -radius]
        speed = self.np_random.uniform(2.0, 3.5)
        color = self.COLOR_APPLE if fruit_type == "apple" else self.COLOR_ORANGE
        self.fruits.append({"pos": pos, "speed": speed, "radius": radius, "type": fruit_type, "color": color})

    def _spawn_bomb(self):
        radius = 18
        pos = [self.np_random.integers(radius, self.WIDTH - radius), -radius]
        speed = self.np_random.uniform(2.5, 4.0)
        self.bombs.append({"pos": pos, "speed": speed, "radius": radius})

    def _update_difficulty(self):
        # Increase fall speed over time
        if self.steps % 100 == 0 and self.steps > 0:
            self.fall_speed_multiplier += 0.05
        # Increase bomb spawn rate (by reducing interval)
        self.base_bomb_spawn_interval = max(0.5, self.base_bomb_spawn_interval - 0.0005)

    def _update_effects(self):
        # Update particles
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.2  # Gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

        # Update explosions
        for e in self.explosions:
            e["radius"] += e["expand_rate"]
            e["life"] -= 1
        self.explosions = [e for e in self.explosions if e["life"] > 0]
        
        # Update slicer flash
        if self.slicer_flash_timer > 0:
            self.slicer_flash_timer -= 1

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_over = True
        if self.bomb_hits >= self.MAX_BOMB_HITS:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bomb_hits": self.bomb_hits,
        }

    def _render_background(self):
        for y in range(self.HEIGHT):
            mix_ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - mix_ratio) + self.COLOR_BG_BOTTOM[0] * mix_ratio),
                int(self.COLOR_BG_TOP[1] * (1 - mix_ratio) + self.COLOR_BG_BOTTOM[1] * mix_ratio),
                int(self.COLOR_BG_TOP[2] * (1 - mix_ratio) + self.COLOR_BG_BOTTOM[2] * mix_ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw fruits
        for fruit in self.fruits:
            pos = (int(fruit["pos"][0]), int(fruit["pos"][1]))
            radius = fruit["radius"]
            if fruit["type"] == "apple":
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_APPLE)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_APPLE)
                pygame.gfxdraw.filled_circle(self.screen, pos[0] - 7, pos[1] - 7, 4, self.COLOR_APPLE_SHINE)
            elif fruit["type"] == "orange":
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ORANGE)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ORANGE)
                pygame.gfxdraw.filled_circle(self.screen, pos[0] - 7, pos[1] - 7, 4, self.COLOR_ORANGE_SHINE)

        # Draw bombs
        for bomb in self.bombs:
            pos = (int(bomb["pos"][0]), int(bomb["pos"][1]))
            radius = bomb["radius"]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_BOMB)
            fuse_end = (pos[0] + 5, pos[1] - (radius + 5))
            pygame.draw.aaline(self.screen, self.COLOR_BOMB_FUSE, (pos[0], pos[1] - radius), fuse_end, 2)
            pygame.gfxdraw.filled_circle(self.screen, fuse_end[0], fuse_end[1], 2, (255, 200, 0))

        # Draw effects
        self._draw_particles()
        self._draw_explosions()

        # Draw slicer
        if self.slicer_flash_timer > 0:
            alpha = 255 * (self.slicer_flash_timer / 3)
            line_color = (*self.COLOR_SLICER_ACTIVE, alpha)
            pygame.draw.line(self.screen, line_color, (self.slicer_x, 0), (self.slicer_x, self.HEIGHT), 4)
        else:
            pygame.draw.line(self.screen, self.COLOR_SLICER, (self.slicer_x, 0), (self.slicer_x, self.HEIGHT), 2)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"{int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Bomb hits display
        for i in range(self.MAX_BOMB_HITS):
            pos = (self.WIDTH - 30 - i * 35, 25)
            color = self.COLOR_BOMB_ICON if i < self.bomb_hits else (80, 80, 80)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, color)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed - 2]
            life = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "life": life, "color": color})

    def _create_explosion(self, pos):
        self.explosions.append({
            "pos": pos,
            "radius": 10,
            "expand_rate": 8,
            "life": 10,
            "color1": (255, 150, 0),
            "color2": (255, 50, 0)
        })

    def _draw_particles(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = max(0, int(p["life"] * 0.2))
            pygame.draw.rect(self.screen, p["color"], (*pos, size, size))

    def _draw_explosions(self):
        for e in self.explosions:
            alpha = int(255 * (e["life"] / 10))
            color1 = (*e["color1"], alpha)
            color2 = (*e["color2"], alpha)
            pos = (int(e["pos"][0]), int(e["pos"][1]))
            radius = int(e["radius"])
            
            # Draw with alpha blending
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(surf, radius, radius, radius, color1)
            pygame.gfxdraw.filled_circle(surf, radius, radius, int(radius * 0.6), color2)
            self.screen.blit(surf, (pos[0] - radius, pos[1] - radius))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Game over check ---
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(2000) # Pause before closing
            
        clock.tick(GameEnv.FPS)
        
    env.close()