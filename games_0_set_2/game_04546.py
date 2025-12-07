
# Generated: 2025-08-28T02:43:47.715467
# Source Brief: brief_04546.md
# Brief Index: 4546

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to place a reinforcing block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your fortress from waves of projectiles by strategically placing reinforcing blocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.BLOCK_SIZE = 20
        self.MAX_STEPS = 2000
        self.MAX_WAVES = 10

        # Game parameters
        self.FORTRESS_HEALTH_INITIAL = 100
        self.BLOCK_HEALTH_INITIAL = 3
        self.BLOCKS_PER_WAVE = 5
        self.INITIAL_PROJECTILES_PER_WAVE = 5
        self.INITIAL_PROJECTILE_SPEED = 1.0
        self.WAVE_COOLDOWN_FRAMES = 90  # 3 seconds at 30fps

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_FORTRESS = (50, 60, 70)
        self.COLOR_CURSOR = (100, 150, 255, 128) # RGBA for transparency
        self.COLOR_PROJECTILE = (255, 80, 80)
        self.COLOR_BLOCK_HEALTH = [
            (80, 120, 255), # Full health
            (70, 100, 220), # 2 HP
            (60, 80, 180),  # 1 HP
        ]
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BAR = (80, 220, 120)
        self.COLOR_HEALTH_BAR_BG = (220, 80, 80)
        self.COLOR_PARTICLE = (255, 200, 150)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
            self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 54)
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.fortress_health = 0
        self.fortress_rect = None
        self.projectiles = []
        self.blocks = {}
        self.particles = []
        self.cursor_pos = [0, 0]
        self.current_wave = 0
        self.blocks_to_place = 0
        self.wave_timer = 0
        self.prev_space_held = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.fortress_health = self.FORTRESS_HEALTH_INITIAL
        fortress_w = self.WIDTH - 8 * self.BLOCK_SIZE
        fortress_h = self.HEIGHT - 8 * self.BLOCK_SIZE
        self.fortress_rect = pygame.Rect(
            (self.WIDTH - fortress_w) // 2,
            (self.HEIGHT - fortress_h) // 2,
            fortress_w,
            fortress_h,
        )

        self.projectiles = []
        self.blocks = {} # Use dict with (x,y) tuple as key for faster lookups
        self.particles = []

        self.cursor_pos = [
            self.fortress_rect.centerx // self.BLOCK_SIZE,
            self.fortress_rect.centery // self.BLOCK_SIZE,
        ]

        self.current_wave = 0
        self.blocks_to_place = self.BLOCKS_PER_WAVE
        self.wave_timer = self.WAVE_COOLDOWN_FRAMES // 2
        self.prev_space_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        terminated = False

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Handle Input ---
        self._handle_input(movement, space_held)

        # --- Update Game Logic ---
        if not self.game_over and not self.win:
            self.steps += 1
            reward -= 0.01 # Survival penalty to encourage speed

            self._update_projectiles()
            reward += self._check_projectile_collisions()
            self._update_particles()
            
            # Wave progression logic
            if self.wave_timer > 0:
                self.wave_timer -= 1
                if self.wave_timer == 0:
                    self._start_new_wave()
            elif not self.projectiles and not self.game_over:
                self.current_wave += 1
                if self.current_wave > self.MAX_WAVES:
                    self.win = True
                else:
                    # sfx: wave_complete
                    reward += 1.0
                    self.blocks_to_place += self.BLOCKS_PER_WAVE
                    self.wave_timer = self.WAVE_COOLDOWN_FRAMES

        # --- Check Termination Conditions ---
        if self.fortress_health <= 0 and not self.game_over:
            self.game_over = True
            terminated = True
            reward = -100.0
            # sfx: fortress_destroyed

        if self.win and not terminated:
            terminated = True
            reward = 100.0
            # sfx: game_win

        if self.steps >= self.MAX_STEPS and not terminated:
            self.game_over = True
            terminated = True
            reward = -100.0 # Timeout is a failure

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held):
        # Move cursor
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        new_cursor_x = self.cursor_pos[0] + dx
        new_cursor_y = self.cursor_pos[1] + dy

        # Clamp cursor to fortress boundaries
        min_x = self.fortress_rect.left // self.BLOCK_SIZE
        max_x = (self.fortress_rect.right // self.BLOCK_SIZE) - 1
        min_y = self.fortress_rect.top // self.BLOCK_SIZE
        max_y = (self.fortress_rect.bottom // self.BLOCK_SIZE) - 1

        self.cursor_pos[0] = max(min_x, min(new_cursor_x, max_x))
        self.cursor_pos[1] = max(min_y, min(new_cursor_y, max_y))

        # Place block on key press (rising edge)
        if space_held and not self.prev_space_held:
            if self.blocks_to_place > 0 and tuple(self.cursor_pos) not in self.blocks:
                # sfx: place_block
                self.blocks[tuple(self.cursor_pos)] = self.BLOCK_HEALTH_INITIAL
                self.blocks_to_place -= 1
        
        self.prev_space_held = space_held

    def _start_new_wave(self):
        num_projectiles = self.INITIAL_PROJECTILES_PER_WAVE + self.current_wave
        speed = self.INITIAL_PROJECTILE_SPEED + self.current_wave * 0.2

        for _ in range(num_projectiles):
            # Spawn from one of the four sides
            side = self.np_random.integers(0, 4)
            if side == 0:  # Top
                pos = [self.np_random.uniform(0, self.WIDTH), -10]
                angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75)
            elif side == 1:  # Bottom
                pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 10]
                angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            elif side == 2:  # Left
                pos = [-10, self.np_random.uniform(0, self.HEIGHT)]
                angle = self.np_random.uniform(-math.pi * 0.25, math.pi * 0.25)
            else:  # Right
                pos = [self.WIDTH + 10, self.np_random.uniform(0, self.HEIGHT)]
                angle = self.np_random.uniform(math.pi * 0.75, math.pi * 1.25)

            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.projectiles.append({"pos": pos, "vel": vel, "radius": 5})

    def _update_projectiles(self):
        for p in self.projectiles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]

    def _check_projectile_collisions(self):
        reward = 0
        projectiles_to_remove = []
        blocks_to_remove = []

        for i, p in enumerate(self.projectiles):
            p_pos = p["pos"]
            p_rect = pygame.Rect(p_pos[0] - p["radius"], p_pos[1] - p["radius"], p["radius"] * 2, p["radius"] * 2)

            # Fortress collision
            if self.fortress_rect.colliderect(p_rect):
                self.fortress_health -= 5
                projectiles_to_remove.append(i)
                self._create_particles(p_pos, 15, self.COLOR_HEALTH_BAR_BG)
                # sfx: fortress_hit
                continue

            # Block collision
            grid_x, grid_y = int(p_pos[0] // self.BLOCK_SIZE), int(p_pos[1] // self.BLOCK_SIZE)
            collided = False
            for dx in [-1, 0, 1]:
                if collided: break
                for dy in [-1, 0, 1]:
                    check_pos = (grid_x + dx, grid_y + dy)
                    if check_pos in self.blocks:
                        block_rect = pygame.Rect(check_pos[0] * self.BLOCK_SIZE, check_pos[1] * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                        if block_rect.colliderect(p_rect):
                            self.blocks[check_pos] -= 1
                            reward += 0.1
                            projectiles_to_remove.append(i)
                            self._create_particles(p_pos, 10, self.COLOR_BLOCK_HEALTH[0])
                            # sfx: projectile_hit
                            if self.blocks[check_pos] <= 0:
                                blocks_to_remove.append(check_pos)
                            collided = True
                            break
            if collided:
                continue

            # Wall collision (despawn)
            if not (-20 < p_pos[0] < self.WIDTH + 20 and -20 < p_pos[1] < self.HEIGHT + 20):
                projectiles_to_remove.append(i)

        for i in sorted(list(set(projectiles_to_remove)), reverse=True):
            del self.projectiles[i]
        
        for pos in set(blocks_to_remove):
            if pos in self.blocks:
                del self.blocks[pos]
                # sfx: block_destroyed

        return reward

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Fortress
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS, self.fortress_rect)

        # Blocks
        for pos, health in self.blocks.items():
            rect = pygame.Rect(pos[0] * self.BLOCK_SIZE, pos[1] * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
            color_index = max(0, health - 1)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK_HEALTH[color_index], rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

        # Cursor
        cursor_surface = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
        cursor_surface.fill(self.COLOR_CURSOR)
        self.screen.blit(cursor_surface, (self.cursor_pos[0] * self.BLOCK_SIZE, self.cursor_pos[1] * self.BLOCK_SIZE))

        # Projectiles
        for p in self.projectiles:
            x, y = int(p["pos"][0]), int(p["pos"][1])
            r = int(p["radius"])
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_PROJECTILE)

        # Particles
        for p in self.particles:
            life_ratio = p["life"] / 30.0
            size = int(2 * life_ratio)
            if size > 0:
                color = (p["color"][0], p["color"][1], p["color"][2], int(255 * life_ratio))
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (int(p["pos"][0] - size), int(p["pos"][1] - size)))

    def _render_ui(self):
        # Health bar
        bar_width = 200
        bar_height = 15
        health_ratio = max(0, self.fortress_health / self.FORTRESS_HEALTH_INITIAL)
        current_health_width = int(bar_width * health_ratio)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, current_health_width, bar_height))
        health_text = self.font_small.render("FORTRESS", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 10 + bar_height))

        # Wave counter
        wave_str = f"WAVE: {min(self.current_wave, self.MAX_WAVES)} / {self.MAX_WAVES}"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Blocks to place
        blocks_str = f"BLOCKS: {self.blocks_to_place}"
        blocks_text = self.font_small.render(blocks_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(blocks_text, (self.WIDTH // 2 - blocks_text.get_width() // 2, self.HEIGHT - 30))

        # Game Over / Win message
        msg, color = (None, None)
        if self.game_over:
            msg, color = "FORTRESS DESTROYED", self.COLOR_HEALTH_BAR_BG
        elif self.win:
            msg, color = "VICTORY", self.COLOR_HEALTH_BAR
        
        if msg:
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "fortress_health": self.fortress_health,
            "blocks_remaining": self.blocks_to_place,
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Fortress Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.set_caption(f"Score: {info['score']:.2f} | Wave: {info['wave']}")
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30)
        
    env.close()