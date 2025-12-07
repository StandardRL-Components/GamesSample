import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:28:21.729711
# Source Brief: brief_02803.md
# Brief Index: 2803
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Defend a cellular wall from rhythmic waves of bacteria.

    This environment simulates a microscopic world where the agent must protect a central
    cell wall from attacking bacteria. The agent controls a magnetic field to guide
    energy bursts and can deploy repair units to mend the wall.

    **Visuals:**
    - A bioluminescent, microscopic aesthetic.
    - The central element is a large, circular cell wall that shows damage locally.
    - Player's actions are visualized through a rotating aiming reticle, bright blue
      energy bursts, and purple repair units.
    - Enemies are pulsating yellow bacteria.
    - UI elements clearly display wall integrity, energy, score, and current wave.

    **Gameplay:**
    - **Objective:** Survive all waves of bacteria. The game ends if the wall's
      average integrity drops to zero or if all waves are cleared.
    - **Actions:**
        1. Rotate the magnetic aimer (influences projectile path).
        2. Fire an energy burst (costs energy).
        3. Deploy a repair unit on the most damaged wall section (costs energy).
    - **Mechanics:**
        - Bacteria spawn in waves and move towards the center.
        - Energy bursts destroy bacteria on contact.
        - Repair units slowly restore the integrity of the wall segment they are on.
        - Energy regenerates slowly over time.

    **Rewards:**
    - Positive rewards for destroying bacteria, repairing the wall, and completing waves.
    - Large terminal rewards for winning or losing the game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your central cell wall from waves of attacking bacteria. Use your magnetic field to aim and fire energy bursts, and deploy repair units to mend the wall."
    )
    user_guide = (
        "Controls: Use ←↑ keys to rotate the aimer counter-clockwise and →↓ keys to rotate it clockwise. Press space to fire an energy burst and shift to deploy a repair unit."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    WALL_RADIUS = 150
    WALL_SEGMENTS = 360
    MAX_STEPS = 2000
    TOTAL_WAVES = 10

    # --- Colors ---
    COLOR_BG = (10, 5, 30)
    COLOR_WALL_HEALTHY = (0, 255, 100)
    COLOR_WALL_DAMAGED = (100, 0, 20)
    COLOR_BURST = (100, 200, 255)
    COLOR_BACTERIA = (255, 255, 0)
    COLOR_REPAIR_UNIT = (200, 0, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BAR = (40, 40, 80)
    COLOR_UI_ENERGY = (0, 150, 255)
    COLOR_UI_INTEGRITY = (0, 200, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wall_integrity = np.full(self.WALL_SEGMENTS, 100.0, dtype=np.float32)
        self.energy = 100.0
        self.magnetic_angle = -math.pi / 2
        self.bacteria = []
        self.bursts = []
        self.repair_units = []
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.current_wave = 0
        self.wave_cooldown = 0
        self.bacteria_in_wave = 0
        self.bacteria_spawn_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.wall_integrity.fill(100.0)
        self.energy = 100.0
        self.magnetic_angle = -math.pi / 2
        
        self.bacteria.clear()
        self.bursts.clear()
        self.repair_units.clear()
        self.particles.clear()

        self.last_space_held = False
        self.last_shift_held = False

        self.current_wave = 0
        self.wave_cooldown = 120 # Start first wave after 4 seconds (at 30fps)
        self.bacteria_in_wave = 0
        self.bacteria_spawn_timer = 0
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        self._handle_input(action)
        reward += self._update_game_state()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            if self.current_wave > self.TOTAL_WAVES:
                reward += 100 # Win bonus
            elif np.mean(self.wall_integrity) <= 0:
                reward -= 100 # Lose penalty
        
        # Assertions for stability
        assert 0 <= self.energy <= 100.1, f"Energy out of bounds: {self.energy}"
        assert 0 <= np.mean(self.wall_integrity) <= 100.1, f"Integrity out of bounds: {np.mean(self.wall_integrity)}"

        obs = self._get_observation()
        info = self._get_info()
        
        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        rotation_speed = math.radians(4)
        if movement in [1, 3]: # Up or Left -> CCW
            self.magnetic_angle -= rotation_speed
        elif movement in [2, 4]: # Down or Right -> CW
            self.magnetic_angle += rotation_speed
        self.magnetic_angle %= (2 * math.pi)

        if space_held and not self.last_space_held:
            self._fire_burst() # sfx: laser_shoot.wav
        
        if shift_held and not self.last_shift_held:
            self._deploy_repair_unit() # sfx: deploy_unit.wav

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_game_state(self):
        step_reward = 0.0
        
        self.energy = min(100.0, self.energy + 0.05) # Slow energy regen

        step_reward += self._update_bursts()
        step_reward += self._update_bacteria()
        step_reward += self._update_repairs()
        self._update_particles()
        step_reward += self._update_wave_manager()
        
        return step_reward

    def _update_bursts(self):
        reward = 0
        for burst in self.bursts[:]:
            burst['pos'][0] += burst['vel'][0]
            burst['pos'][1] += burst['vel'][1]
            burst['ttl'] -= 1
            if burst['ttl'] <= 0:
                self.bursts.remove(burst)
                continue
            
            for bacterium in self.bacteria[:]:
                dist = math.hypot(burst['pos'][0] - bacterium['pos'][0], burst['pos'][1] - bacterium['pos'][1])
                if dist < bacterium['size']:
                    self._create_particles(bacterium['pos'], self.COLOR_BACTERIA, 15) # sfx: bacteria_die.wav
                    self.bacteria.remove(bacterium)
                    if burst in self.bursts:
                        self.bursts.remove(burst)
                    self.score += 10
                    reward += 0.1
                    self.bacteria_in_wave -= 1
                    break
        return reward

    def _update_bacteria(self):
        for bacterium in self.bacteria[:]:
            # Move towards center
            dx, dy = self.CENTER[0] - bacterium['pos'][0], self.CENTER[1] - bacterium['pos'][1]
            dist = math.hypot(dx, dy)
            if dist == 0: continue
            
            bacterium['pos'][0] += dx / dist * bacterium['speed']
            bacterium['pos'][1] += dy / dist * bacterium['speed']
            
            # Pulsate
            bacterium['pulse'] += 0.1
            bacterium['size'] = bacterium['base_size'] + math.sin(bacterium['pulse']) * 2

            # Check collision with wall
            if dist <= self.WALL_RADIUS:
                angle = math.atan2(dy, dx)
                segment_idx = int((angle + math.pi) / (2 * math.pi) * self.WALL_SEGMENTS) % self.WALL_SEGMENTS
                self.wall_integrity[segment_idx] = max(0, self.wall_integrity[segment_idx] - 25)
                self._create_particles(bacterium['pos'], self.COLOR_WALL_DAMAGED, 10, 1) # sfx: wall_hit.wav
                self.bacteria.remove(bacterium)
                self.bacteria_in_wave -= 1
        return 0

    def _update_repairs(self):
        reward = 0
        for unit in self.repair_units[:]:
            integrity_before = self.wall_integrity[unit['segment']]
            self.wall_integrity[unit['segment']] = min(100, integrity_before + 0.5)
            healed_amount = self.wall_integrity[unit['segment']] - integrity_before
            reward += healed_amount * 0.01 # Reward for healing
            self.score += healed_amount * 0.1

            unit['ttl'] -= 1
            if unit['ttl'] <= 0 or self.wall_integrity[unit['segment']] >= 100:
                self.repair_units.remove(unit)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['ttl'] -= 1
            if p['ttl'] <= 0:
                self.particles.remove(p)

    def _update_wave_manager(self):
        reward = 0
        if self.bacteria_in_wave <= 0 and self.current_wave <= self.TOTAL_WAVES:
            if self.wave_cooldown > 0:
                self.wave_cooldown -= 1
            else:
                self.current_wave += 1
                if self.current_wave > self.TOTAL_WAVES: return 0 # Game won
                
                # sfx: wave_start.wav
                wave_data = self._get_wave_data(self.current_wave)
                self.bacteria_in_wave = wave_data['count']
                self.bacteria_spawn_timer = wave_data['delay']
                self.wave_cooldown = 180 # 6 seconds between waves
                reward += 1.0
                self.score += 100
        
        if self.bacteria_in_wave > 0 and self.bacteria_spawn_timer > 0:
            self.bacteria_spawn_timer -= 1
            if self.bacteria_spawn_timer <= 0:
                wave_data = self._get_wave_data(self.current_wave)
                self._spawn_bacterium(wave_data['speed'])
                self.bacteria_spawn_timer = wave_data['delay']
        return reward

    def _get_wave_data(self, wave_num):
        base_count = 5
        base_speed = 1.0
        base_delay = 30 # 1 second
        
        # Difficulty scaling: increases every few levels
        difficulty_tier = (wave_num -1) // 3
        count = base_count + difficulty_tier * 2
        speed = base_speed + difficulty_tier * 0.2
        delay = max(5, base_delay - difficulty_tier * 5)
        
        return {'count': count, 'speed': speed, 'delay': delay}

    def _spawn_bacterium(self, speed):
        angle = self.np_random.uniform(0, 2 * math.pi)
        spawn_dist = max(self.SCREEN_WIDTH, self.SCREEN_HEIGHT) / 2 + 20
        pos = [
            self.CENTER[0] + spawn_dist * math.cos(angle),
            self.CENTER[1] + spawn_dist * math.sin(angle)
        ]
        self.bacteria.append({
            'pos': pos, 'speed': speed, 'base_size': self.np_random.uniform(8, 12),
            'size': 10, 'pulse': 0
        })

    def _fire_burst(self):
        cost = 10
        if self.energy >= cost:
            self.energy -= cost
            speed = 8
            vel = [speed * math.cos(self.magnetic_angle), speed * math.sin(self.magnetic_angle)]
            self.bursts.append({
                'pos': list(self.CENTER), 'vel': vel, 'ttl': 80
            })

    def _deploy_repair_unit(self):
        cost = 25
        if self.energy >= cost and np.min(self.wall_integrity) < 100:
            self.energy -= cost
            target_segment = np.argmin(self.wall_integrity)
            
            # Check if a unit is already there
            for unit in self.repair_units:
                if unit['segment'] == target_segment:
                    unit['ttl'] = 150 # Refresh existing unit
                    return

            angle = (target_segment / self.WALL_SEGMENTS) * 2 * math.pi - math.pi
            pos = [
                self.CENTER[0] + self.WALL_RADIUS * math.cos(angle),
                self.CENTER[1] + self.WALL_RADIUS * math.sin(angle)
            ]
            self.repair_units.append({
                'pos': pos, 'segment': target_segment, 'ttl': 150
            })

    def _create_particles(self, pos, color, count, speed_mult=3):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'ttl': self.np_random.integers(10, 20),
                'color': color,
                'size': self.np_random.uniform(1, 3)
            })

    def _check_termination(self):
        if np.mean(self.wall_integrity) <= 0:
            return True
        if self.current_wave > self.TOTAL_WAVES and not self.bacteria:
            return True
        return False

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
            "wave": self.current_wave,
            "energy": self.energy,
            "wall_integrity": np.mean(self.wall_integrity)
        }
    
    def _render_game(self):
        self._render_magnetic_field()
        self._render_wall()
        
        for unit in self.repair_units:
            self._draw_glow_circle(self.screen, self.COLOR_REPAIR_UNIT, unit['pos'], 8, 3)
            pygame.gfxdraw.filled_circle(self.screen, int(unit['pos'][0]), int(unit['pos'][1]), 6, self.COLOR_REPAIR_UNIT)

        for burst in self.bursts:
            self._draw_glow_circle(self.screen, self.COLOR_BURST, burst['pos'], 5, 4)
            pygame.gfxdraw.filled_circle(self.screen, int(burst['pos'][0]), int(burst['pos'][1]), 3, (255,255,255))
        
        for bacterium in self.bacteria:
            self._draw_glow_circle(self.screen, self.COLOR_BACTERIA, bacterium['pos'], bacterium['size'] + 2, 3)
            pygame.gfxdraw.filled_circle(self.screen, int(bacterium['pos'][0]), int(bacterium['pos'][1]), int(bacterium['size']), self.COLOR_BACTERIA)

        for p in self.particles:
            alpha = max(0, 255 * (p['ttl'] / 20))
            color = (*p['color'], alpha)
            if len(p['color']) == 4: # Already has alpha
                color = (*p['color'][:3], alpha)
            
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

    def _render_wall(self):
        for i in range(self.WALL_SEGMENTS):
            health_ratio = self.wall_integrity[i] / 100.0
            color = (
                int(self.COLOR_WALL_DAMAGED[0] + (self.COLOR_WALL_HEALTHY[0] - self.COLOR_WALL_DAMAGED[0]) * health_ratio),
                int(self.COLOR_WALL_DAMAGED[1] + (self.COLOR_WALL_HEALTHY[1] - self.COLOR_WALL_DAMAGED[1]) * health_ratio),
                int(self.COLOR_WALL_DAMAGED[2] + (self.COLOR_WALL_HEALTHY[2] - self.COLOR_WALL_DAMAGED[2]) * health_ratio),
            )
            
            angle_start = (i / self.WALL_SEGMENTS) * 2 * math.pi
            angle_end = ((i + 1) / self.WALL_SEGMENTS) * 2 * math.pi
            
            p1 = (self.CENTER[0] + self.WALL_RADIUS * math.cos(angle_start), self.CENTER[1] + self.WALL_RADIUS * math.sin(angle_start))
            p2 = (self.CENTER[0] + self.WALL_RADIUS * math.cos(angle_end), self.CENTER[1] + self.WALL_RADIUS * math.sin(angle_end))
            p3 = (self.CENTER[0] + (self.WALL_RADIUS - 6) * math.cos(angle_end), self.CENTER[1] + (self.WALL_RADIUS - 6) * math.sin(angle_end))
            p4 = (self.CENTER[0] + (self.WALL_RADIUS - 6) * math.cos(angle_start), self.CENTER[1] + (self.WALL_RADIUS - 6) * math.sin(angle_start))
            
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], color)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], color)

    def _render_magnetic_field(self):
        # Render aiming reticle
        aim_dist = self.WALL_RADIUS + 20
        end_pos = (
            self.CENTER[0] + aim_dist * math.cos(self.magnetic_angle),
            self.CENTER[1] + aim_dist * math.sin(self.magnetic_angle)
        )
        pygame.draw.aaline(self.screen, (255, 255, 255, 50), self.CENTER, end_pos, 1)
        pygame.gfxdraw.filled_circle(self.screen, int(end_pos[0]), int(end_pos[1]), 5, (255, 255, 255, 150))
        pygame.gfxdraw.aacircle(self.screen, int(end_pos[0]), int(end_pos[1]), 5, (255, 255, 255, 150))

    def _render_ui(self):
        # --- UI Panel Background ---
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill((0, 0, 0, 100))
        self.screen.blit(ui_panel, (0, 0))

        # --- Wall Integrity ---
        integrity_text = self.font_small.render("Integrity", True, self.COLOR_UI_TEXT)
        self.screen.blit(integrity_text, (10, 12))
        avg_integrity = np.mean(self.wall_integrity)
        self._draw_bar(100, 10, 150, 20, avg_integrity / 100.0, self.COLOR_UI_INTEGRITY)

        # --- Energy ---
        energy_text = self.font_small.render("Energy", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (260, 12))
        self._draw_bar(320, 10, 150, 20, self.energy / 100.0, self.COLOR_UI_ENERGY)

        # --- Score & Wave ---
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 8))
        
        wave_str = f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}"
        if self.current_wave > self.TOTAL_WAVES: wave_str = "VICTORY!"
        if self.game_over and np.mean(self.wall_integrity) <= 0: wave_str = "GAME OVER"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (480, 12))

    def _draw_bar(self, x, y, w, h, progress, color):
        progress = max(0, min(1, progress))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (x, y, w, h))
        pygame.draw.rect(self.screen, color, (x, y, int(w * progress), h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (x, y, w, h), 1)

    def _draw_glow_circle(self, surface, color, pos, radius, intensity):
        """Draws a circle with a glowing effect."""
        pos = (int(pos[0]), int(pos[1]))
        for i in range(intensity):
            alpha = 150 - (i * (150 // intensity))
            r = radius + i * 2
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(r), (*color, alpha))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a window and render the game
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cellular Defense")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    truncated = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        if terminated or truncated:
            # Game over screen
            obs = env._get_observation()
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            font = pygame.font.SysFont("Consolas", 48, bold=True)
            text = "GAME OVER"
            if env.current_wave > env.TOTAL_WAVES:
                text = "VICTORY!"
            
            text_surf = font.render(text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=env.CENTER)
            screen.blit(text_surf, text_rect)
            
            font_small = pygame.font.SysFont("Consolas", 24)
            restart_text = font_small.render("Press 'R' to restart", True, (200, 200, 200))
            restart_rect = restart_text.get_rect(center=(env.CENTER[0], env.CENTER[1] + 50))
            screen.blit(restart_text, restart_rect)
            
            pygame.display.flip()
            continue

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        move_action = 0 # None
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation to pygame surface and draw
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()