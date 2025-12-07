import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to place block. Shift to cycle block type."
    )

    game_description = (
        "Defend your procedurally generated fortress against waves of enemies by strategically placing defensive blocks."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_WAVES = 20
    MAX_STEPS = 15000 # Approx 20 waves * (5s place + 20s wave) * 30fps

    COLOR_BG = (20, 25, 30)
    COLOR_WALL = (60, 70, 80)
    COLOR_BASE = (0, 200, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BAR_BG = (80, 20, 20)

    BLOCK_STATS = {
        0: {"color": (50, 100, 255), "health": 50, "cost": 0.01},  # Blue
        1: {"color": (255, 200, 50), "health": 100, "cost": 0.02}, # Yellow
        2: {"color": (200, 50, 255), "health": 200, "cost": 0.04}, # Purple
    }

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        self.game_phase = "placement"
        self.placement_timer = 0
        self.wave_number = 0
        self.base_health = 0
        self.base_max_health = 100
        self.base_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT - 30)
        self.base_rect = None
        self.fortress_walls = []
        self.blocks = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.grid_size = 20
        self.selected_block_type = 0
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        # This will be initialized in reset()
        self.np_random = None
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.game_phase = "placement"
        self.placement_timer = 5 * self.FPS # 5 seconds
        self.wave_number = 1
        self.base_health = self.base_max_health
        self.base_rect = pygame.Rect(self.base_pos.x - 20, self.base_pos.y - 20, 40, 40)
        
        self.fortress_walls = self._generate_fortress()
        self.blocks = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.selected_block_type = 0
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.space_pressed_last_frame
        shift_press = shift_held and not self.shift_pressed_last_frame

        if self.game_phase == "placement":
            reward += self._update_placement_phase(movement, space_press, shift_press)
        elif self.game_phase == "wave":
            wave_reward, wave_ended = self._update_wave_phase()
            reward += wave_reward
            if wave_ended:
                if self.wave_number >= self.MAX_WAVES:
                    self.game_over = True
                    reward += 50 # Win bonus
                else:
                    reward += 1 # Wave survival bonus
                    self.wave_number += 1
                    self.game_phase = "placement"
                    self.placement_timer = 5 * self.FPS
        
        self._update_particles()
        
        # Don't add reward to score directly, as terminal rewards are handled specially
        current_step_reward = reward
        
        self.space_pressed_last_frame = space_held
        self.shift_pressed_last_frame = shift_held

        terminated = self.base_health <= 0 or self.game_over or self.steps >= self.MAX_STEPS
        if terminated:
            if self.base_health <= 0 and not self.game_over:
                 current_step_reward = -50 # Loss penalty
                 self.game_over = True
            self.score += current_step_reward # Add final reward to score
        else:
            self.score += current_step_reward
        
        return self._get_observation(), current_step_reward, terminated, False, self._get_info()

    def _update_placement_phase(self, movement, space_press, shift_press):
        reward = 0
        self.placement_timer -= 1

        # Move cursor
        cursor_speed = 8
        if movement == 1: self.cursor_pos.y -= cursor_speed
        elif movement == 2: self.cursor_pos.y += cursor_speed
        elif movement == 3: self.cursor_pos.x -= cursor_speed
        elif movement == 4: self.cursor_pos.x += cursor_speed
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)
        
        # Cycle block type
        if shift_press:
            self.selected_block_type = (self.selected_block_type + 1) % len(self.BLOCK_STATS)
            # sfx: UI_Cycle.wav
        
        # Place block
        if space_press:
            grid_x = round(self.cursor_pos.x / self.grid_size) * self.grid_size
            grid_y = round(self.cursor_pos.y / self.grid_size) * self.grid_size
            new_block_rect = pygame.Rect(grid_x - self.grid_size//2, grid_y - self.grid_size//2, self.grid_size, self.grid_size)
            
            is_valid_placement = True
            if new_block_rect.colliderect(self.base_rect): is_valid_placement = False
            if any(new_block_rect.colliderect(b['rect']) for b in self.blocks): is_valid_placement = False
            if any(new_block_rect.colliderect(w) for w in self.fortress_walls): is_valid_placement = False
            if not (0 <= new_block_rect.left and new_block_rect.right <= self.WIDTH and 0 <= new_block_rect.top and new_block_rect.bottom <= self.HEIGHT - 60):
                is_valid_placement = False

            if is_valid_placement:
                stats = self.BLOCK_STATS[self.selected_block_type]
                self.blocks.append({
                    "rect": new_block_rect,
                    "type": self.selected_block_type,
                    "health": stats["health"],
                    "max_health": stats["health"],
                })
                reward -= stats["cost"]
                # sfx: Block_Place.wav

        if self.placement_timer <= 0:
            self.game_phase = "wave"
            self._spawn_next_wave()
            # sfx: Wave_Start.wav
        
        return reward

    def _update_wave_phase(self):
        reward = 0
        self._update_enemies()
        self._update_projectiles()
        reward += self._handle_collisions()
        wave_ended = len(self.enemies) == 0 and len(self.projectiles) == 0
        return reward, wave_ended

    def _spawn_next_wave(self):
        num_enemies = 3 + self.wave_number
        speed = 1.0 + (self.wave_number // 2) * 0.1
        health = 10 + (self.wave_number // 3) * 5
        fire_rate = max(30, 120 - self.wave_number * 3)
        
        for _ in range(num_enemies):
            spawn_x = self.np_random.integers(20, self.WIDTH - 20)
            spawn_y = self.np_random.integers(20, 60)
            self.enemies.append({
                "pos": pygame.Vector2(spawn_x, spawn_y),
                "health": health,
                "max_health": health,
                "speed": speed,
                "fire_cooldown": self.np_random.integers(fire_rate, fire_rate * 2),
                "fire_rate": fire_rate,
                "size": 8,
            })
    
    def _update_enemies(self):
        obstacles = [b['rect'] for b in self.blocks] + self.fortress_walls
        
        for enemy in self.enemies:
            # Pathfinding
            direction = (self.base_pos - enemy['pos'])
            if direction.length() > 0:
                direction.normalize_ip()

            probe_pos = enemy['pos'] + direction * (enemy['size'] + 2)
            probe_rect = pygame.Rect(probe_pos.x - 2, probe_pos.y - 2, 4, 4)

            if probe_rect.collidelist(obstacles) != -1:
                # Simple avoidance: try turning
                left_dir = direction.rotate(-45)
                right_dir = direction.rotate(45)
                if pygame.Rect(enemy['pos'] + left_dir * 5 - (2,2), (4,4)).collidelist(obstacles) == -1:
                    direction = left_dir
                elif pygame.Rect(enemy['pos'] + right_dir * 5 - (2,2), (4,4)).collidelist(obstacles) == -1:
                    direction = right_dir

            enemy['pos'] += direction * enemy['speed']

            # Shooting
            enemy['fire_cooldown'] -= 1
            if enemy['fire_cooldown'] <= 0:
                proj_dir = (self.base_pos - enemy['pos'])
                if proj_dir.length() > 0:
                    proj_dir.normalize_ip()
                self.projectiles.append({
                    "pos": enemy['pos'].copy(),
                    "vel": proj_dir * 4,
                    "damage": 1 + self.wave_number // 2,
                    "size": 3,
                })
                enemy['fire_cooldown'] = enemy['fire_rate']
                # sfx: Enemy_Shoot.wav

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']
            if not (0 <= proj['pos'].x <= self.WIDTH and 0 <= proj['pos'].y <= self.HEIGHT):
                self.projectiles.remove(proj)

    def _handle_collisions(self):
        reward = 0
        # Enemy projectiles hitting things
        for proj in self.projectiles[:]:
            proj_rect = pygame.Rect(proj['pos'].x - proj['size'], proj['pos'].y - proj['size'], proj['size']*2, proj['size']*2)
            
            # Hit base
            if self.base_rect.colliderect(proj_rect):
                self.base_health -= proj['damage']
                self.projectiles.remove(proj)
                self._create_particles(proj['pos'], self.COLOR_BASE, 10)
                # sfx: Base_Hit.wav
                continue

            # Hit block
            hit_block = False
            for block in self.blocks[:]:
                if block['rect'].colliderect(proj_rect):
                    block['health'] -= proj['damage']
                    reward += 0.1 # Reward for block absorbing a shot
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    self._create_particles(proj['pos'], self.BLOCK_STATS[block['type']]['color'], 5)
                    # sfx: Block_Hit.wav
                    if block['health'] <= 0:
                        self.blocks.remove(block)
                        self._create_particles(block['rect'].center, self.COLOR_WALL, 20)
                        # sfx: Block_Destroy.wav
                    hit_block = True
                    break
            if hit_block:
                continue
        
        # Enemies hitting base
        for enemy in self.enemies[:]:
            enemy_rect = pygame.Rect(enemy['pos'].x - enemy['size'], enemy['pos'].y - enemy['size'], enemy['size']*2, enemy['size']*2)
            if self.base_rect.colliderect(enemy_rect):
                self.base_health -= 20 # High damage for contact
                self.enemies.remove(enemy)
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 30)
                # sfx: Base_Hit_Major.wav
        
        return reward

    def _generate_fortress(self):
        walls = []
        margin = 40
        play_area = pygame.Rect(margin, margin, self.WIDTH - margin*2, self.HEIGHT - margin*2 - 60)

        num_walls = self.np_random.integers(3, 6)
        for _ in range(num_walls):
            is_horizontal = self.np_random.choice([True, False])
            if is_horizontal:
                w = self.np_random.integers(50, 200)
                h = 8
                x = self.np_random.integers(play_area.left, play_area.right - w)
                y = self.np_random.integers(play_area.top, play_area.bottom - h)
            else:
                w = 8
                h = self.np_random.integers(50, 150)
                x = self.np_random.integers(play_area.left, play_area.right - w)
                y = self.np_random.integers(play_area.top, play_area.bottom - h)
            walls.append(pygame.Rect(x, y, w, h))
        return walls

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

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
            "wave": self.wave_number,
            "base_health": self.base_health,
            "enemies_remaining": len(self.enemies),
            "game_phase": self.game_phase,
        }

    def _render_game(self):
        # Walls
        for wall in self.fortress_walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Base
        if self.base_rect:
            pygame.gfxdraw.box(self.screen, self.base_rect, self.COLOR_BASE)
        
        # Blocks
        for block in self.blocks:
            color = self.BLOCK_STATS[block['type']]['color']
            pygame.draw.rect(self.screen, color, block['rect'])
            health_ratio = block['health'] / block['max_health']
            if health_ratio < 1.0:
                border_color = tuple(c * health_ratio for c in color)
                pygame.draw.rect(self.screen, border_color, block['rect'], 2)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            size = int(enemy['size'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY)

        # Projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            size = int(proj['size'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_PROJECTILE)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color_with_alpha = p['color'] + (alpha,)
            size = int(p['life'] / 10)
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color_with_alpha, (size, size), size)
                self.screen.blit(s, (int(p['pos'].x - size), int(p['pos'].y - size)))

    def _render_ui(self):
        # Top bar info
        self._draw_text(f"Wave: {self.wave_number}/{self.MAX_WAVES}", (10, 10), self.font_medium)
        self._draw_text(f"Score: {int(self.score)}", (self.WIDTH - 120, 10), self.font_medium)
        
        # Base health bar
        health_ratio = max(0, self.base_health / self.base_max_health)
        bar_width = 200
        bar_height = 20
        bar_x = (self.WIDTH - bar_width) // 2
        bar_y = self.HEIGHT - 35
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, bar_width * health_ratio, bar_height))
        self._draw_text(f"Base Health", (bar_x + bar_width // 2, bar_y - 15), self.font_small, align="center")

        if self.game_phase == "placement":
            # Placement timer bar
            timer_ratio = max(0, self.placement_timer / (5 * self.FPS))
            pygame.draw.rect(self.screen, (100,100,100), (0, 0, self.WIDTH, 5))
            pygame.draw.rect(self.screen, (200,200,50), (0, 0, self.WIDTH * timer_ratio, 5))

            # Cursor and selected block
            stats = self.BLOCK_STATS[self.selected_block_type]
            cursor_color = stats['color'] + (100,) # with alpha
            s = pygame.Surface((self.grid_size, self.grid_size), pygame.SRCALPHA)
            pygame.draw.rect(s, cursor_color, s.get_rect())
            grid_x = round(self.cursor_pos.x / self.grid_size) * self.grid_size
            grid_y = round(self.cursor_pos.y / self.grid_size) * self.grid_size
            self.screen.blit(s, (grid_x - self.grid_size//2, grid_y - self.grid_size//2))
            
            # Selected block info
            self._draw_text("Selected Block:", (10, self.HEIGHT - 40), self.font_small)
            pygame.draw.rect(self.screen, stats['color'], (120, self.HEIGHT - 40, 20, 20))
            self._draw_text(f"HP: {stats['health']}", (150, self.HEIGHT - 40), self.font_small)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            message = "YOU WIN!" if self.base_health > 0 else "GAME OVER"
            self._draw_text(message, (self.WIDTH//2, self.HEIGHT//2 - 30), self.font_large, align="center")
            self._draw_text(f"Final Score: {int(self.score)}", (self.WIDTH//2, self.HEIGHT//2 + 20), self.font_medium, align="center")

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, align="left"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "right":
            text_rect.topright = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset: This must be done before _get_observation is called,
        # as _get_observation relies on state initialized in reset().
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test observation space (now that the environment is reset)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == "__main__":
    # The main script needs a display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Fortress Defender")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Action mapping for human input
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Render the final frame and wait before resetting
            frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            pygame.time.wait(3000)

            obs, info = env.reset()

        # Render the observation from the environment
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)
        
    pygame.quit()