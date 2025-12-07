
# Generated: 2025-08-27T13:01:53.123879
# Source Brief: brief_00238.md
# Brief Index: 238

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to aim reticle. Space to fire. Shift to retreat to center."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of zombies in an isometric arena. Manage your ammo and position to stay alive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1500 # Increased for longer potential games
        self.WIN_WAVE = 5
        self.PLAYER_MAX_HEALTH = 100
        self.ZOMBIES_PER_WAVE = 20
        self.AMMO_PER_CRATE = 10
        self.RETICLE_SPEED = 8
        self.PROJECTILE_SPEED = 12
        self.PLAYER_RETREAT_SPEED = 1.5

        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_ARENA = (60, 60, 70)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ZOMBIE = (100, 140, 80)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_AMMO_CRATE = (160, 110, 70)
        self.COLOR_BLOOD = (200, 0, 0)
        self.COLOR_RETICLE = (255, 255, 255, 150)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR_BG = (120, 0, 0)
        self.COLOR_HEALTH_BAR_FG = (0, 200, 0)
        
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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables to prevent attribute errors
        self.player_pos = [0, 0]
        self.player_health = 0
        self.ammo = 0
        self.reticle_pos = [0, 0]
        self.zombies = []
        self.projectiles = []
        self.ammo_crates = []
        self.particles = []
        self.wave_number = 0
        self.zombie_speed = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.last_space_held = False
        self.muzzle_flash_timer = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.player_health = self.PLAYER_MAX_HEALTH
        self.ammo = 20
        self.reticle_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        
        self.zombies = []
        self.projectiles = []
        self.ammo_crates = []
        self.particles = []
        
        self.wave_number = 1
        self.zombie_speed = 1.0
        self._spawn_wave()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.last_space_held = False
        self.muzzle_flash_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Handle Player Actions ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Update Game Logic ---
        self._update_projectiles()
        self._update_zombies()
        self._update_player_state(shift_held)
        self._update_particles()
        
        # --- Calculate Rewards ---
        reward += self._calculate_rewards()

        # --- Check Wave Completion ---
        if not self.zombies and not self.game_over:
            reward += 5.0  # Wave clear bonus
            self.wave_number += 1
            if self.wave_number > self.WIN_WAVE:
                self.game_won = True
                self.game_over = True
            else:
                self.zombie_speed += 0.2
                self._spawn_wave()
                self._spawn_ammo_crate()

        self.steps += 1
        self.score += reward
        
        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        if terminated:
            if self.game_won:
                reward += 100.0
            elif self.player_health <= 0:
                reward -= 100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Reticle Movement
        if movement == 1: self.reticle_pos[1] -= self.RETICLE_SPEED
        if movement == 2: self.reticle_pos[1] += self.RETICLE_SPEED
        if movement == 3: self.reticle_pos[0] -= self.RETICLE_SPEED
        if movement == 4: self.reticle_pos[0] += self.RETICLE_SPEED
        self.reticle_pos[0] = np.clip(self.reticle_pos[0], 0, self.WIDTH)
        self.reticle_pos[1] = np.clip(self.reticle_pos[1], 0, self.HEIGHT)

        # Shooting (on key press)
        if space_held and not self.last_space_held:
            if self.ammo > 0:
                self.ammo -= 1
                dx = self.reticle_pos[0] - self.player_pos[0]
                dy = self.reticle_pos[1] - self.player_pos[1]
                dist = math.hypot(dx, dy)
                if dist > 0:
                    vel_x = (dx / dist) * self.PROJECTILE_SPEED
                    vel_y = (dy / dist) * self.PROJECTILE_SPEED
                    self.projectiles.append({
                        "pos": list(self.player_pos),
                        "vel": [vel_x, vel_y]
                    })
                    self.muzzle_flash_timer = 3 # frames
                    # Sound effect placeholder: # pew!
            else:
                # No direct penalty here, miss penalty handles it
                pass 
        self.last_space_held = space_held

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            if not (0 < p["pos"][0] < self.WIDTH and 0 < p["pos"][1] < self.HEIGHT):
                self.projectiles.remove(p)
                # Missed shot penalty is applied in _calculate_rewards

    def _update_zombies(self):
        for z in self.zombies:
            dx = self.player_pos[0] - z["pos"][0]
            dy = self.player_pos[1] - z["pos"][1]
            dist = math.hypot(dx, dy)
            if dist > 0:
                z["pos"][0] += (dx / dist) * self.zombie_speed
                z["pos"][1] += (dy / dist) * self.zombie_speed
            
            # Zombie-Player collision
            if dist < 15: # Collision radius
                self.player_health -= 5
                # Simple knockback
                self.player_pos[0] -= dx / dist * 5
                self.player_pos[1] -= dy / dist * 5
                # Sound effect placeholder: # player_hit

    def _update_player_state(self, shift_held):
        # Retreat action
        if shift_held:
            center_x, center_y = self.WIDTH / 2, self.HEIGHT / 2
            dx = center_x - self.player_pos[0]
            dy = center_y - self.player_pos[1]
            dist = math.hypot(dx, dy)
            if dist > 1:
                self.player_pos[0] += (dx / dist) * self.PLAYER_RETREAT_SPEED
                self.player_pos[1] += (dy / dist) * self.PLAYER_RETREAT_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        self.player_health = max(0, self.player_health)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _calculate_rewards(self):
        reward = 0
        
        # Projectile-Zombie collision
        for p in self.projectiles[:]:
            hit = False
            for z in self.zombies[:]:
                if math.hypot(p["pos"][0] - z["pos"][0], p["pos"][1] - z["pos"][1]) < 12:
                    self.zombies.remove(z)
                    self._create_blood_splatter(z["pos"])
                    reward += 0.1 # Zombie hit
                    hit = True
                    # Sound effect placeholder: # splat!
                    break # Projectile can only hit one zombie
            if hit:
                self.projectiles.remove(p)

        # Missed shot penalty (projectile went off-screen)
        for p in self.projectiles[:]:
            if not (0 < p["pos"][0] < self.WIDTH and 0 < p["pos"][1] < self.HEIGHT):
                reward -= 0.01

        # Player-Ammo Crate collision
        for c in self.ammo_crates[:]:
            if math.hypot(self.player_pos[0] - c[0], self.player_pos[1] - c[1]) < 20:
                self.ammo_crates.remove(c)
                self.ammo += self.AMMO_PER_CRATE
                reward += 1.0 # Ammo pickup
                # Sound effect placeholder: # ammo_pickup
        
        return reward

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        if self.wave_number > self.WIN_WAVE:
            self.game_over = True
            self.game_won = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _spawn_wave(self):
        for _ in range(self.ZOMBIES_PER_WAVE):
            pos = self._get_safe_spawn_pos()
            self.zombies.append({"pos": pos})

    def _spawn_ammo_crate(self):
        pos = self._get_safe_spawn_pos()
        self.ammo_crates.append(pos)

    def _get_safe_spawn_pos(self):
        while True:
            side = self.np_random.integers(4)
            if side == 0: # Top
                pos = [self.np_random.uniform(0, self.WIDTH), -20]
            elif side == 1: # Bottom
                pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20]
            elif side == 2: # Left
                pos = [-20, self.np_random.uniform(0, self.HEIGHT)]
            else: # Right
                pos = [self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT)]
            
            # Ensure it's not too close to the player
            if math.hypot(pos[0] - self.player_pos[0], pos[1] - self.player_pos[1]) > 100:
                return pos

    def _create_blood_splatter(self, pos):
        for _ in range(15): # Number of particles
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 20),
                "radius": self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Arena floor
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (20, 20, self.WIDTH - 40, self.HEIGHT - 40))

        # Ammo crates
        for crate_pos in self.ammo_crates:
            pygame.draw.rect(self.screen, self.COLOR_AMMO_CRATE, (int(crate_pos[0]-10), int(crate_pos[1]-10), 20, 20))

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, self.COLOR_BLOOD, (int(p["pos"][0]), int(p["pos"][1])), int(p["radius"]))
            
        # Zombies
        for z in self.zombies:
            pos = (int(z["pos"][0]), int(z["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_ZOMBIE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_ZOMBIE)

        # Player glow
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), 18, self.COLOR_PLAYER_GLOW)
        
        # Player (triangle)
        p1 = (self.player_pos[0], self.player_pos[1] - 12)
        p2 = (self.player_pos[0] - 10, self.player_pos[1] + 8)
        p3 = (self.player_pos[0] + 10, self.player_pos[1] + 8)
        # Rotate triangle to face reticle
        angle = math.atan2(self.reticle_pos[1] - self.player_pos[1], self.reticle_pos[0] - self.player_pos[0]) - math.pi / 2
        center = self.player_pos
        points = [p1, p2, p3]
        rotated_points = [
            (
                center[0] + (x - center[0]) * math.cos(angle) - (y - center[1]) * math.sin(angle),
                center[1] + (x - center[0]) * math.sin(angle) + (y - center[1]) * math.cos(angle)
            ) for x, y in points
        ]
        pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x, y in rotated_points], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in rotated_points], self.COLOR_PLAYER)

        # Muzzle Flash
        if self.muzzle_flash_timer > 0:
            self.muzzle_flash_timer -= 1
            flash_pos = (int(rotated_points[0][0]), int(rotated_points[0][1]))
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, flash_pos, 8)

        # Projectiles
        for p in self.projectiles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, pos, 3)

        # Reticle
        rx, ry = int(self.reticle_pos[0]), int(self.reticle_pos[1])
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx - 10, ry), (rx + 10, ry), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx, ry - 10), (rx, ry + 10), 2)

    def _render_ui(self):
        # Health Bar
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_pct), 20))

        # Ammo Count
        ammo_text = self.font_small.render(f"AMMO: {self.ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (220, 12))

        # Wave Number
        wave_text = self.font_large.render(f"WAVE {self.wave_number}", True, self.COLOR_UI_TEXT)
        text_rect = wave_text.get_rect(center=(self.WIDTH // 2, 30))
        self.screen.blit(wave_text, text_rect)

        # Game Over / Win message
        if self.game_over:
            message = "YOU SURVIVED!" if self.game_won else "GAME OVER"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_BLOOD
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            # Add a semi-transparent background for readability
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.player_health,
            "ammo": self.ammo,
            "zombies_left": len(self.zombies),
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

# Example of how to run the environment
if __name__ == "__main__":
    import os
    # Set a display for Pygame, or use a dummy driver if no display is available
    if os.environ.get("DISPLAY") is None:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # This requires a display. Run this part on a local machine.
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Zombie Survival")
        
        obs, info = env.reset()
        terminated = False
        running = True
        
        while running and not terminated:
            # Map keyboard keys to the MultiDiscrete action space
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation to the display window
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
            
            env.clock.tick(30) # Limit to 30 FPS
            
        print(f"Game Over. Final Score: {info['score']:.2f}, Wave: {info['wave']}")
        pygame.time.wait(2000) # Pause before closing
        env.close()

    # --- Agent Interaction Example ---
    # This can run anywhere
    else:
        print("\nRunning headless agent interaction example...")
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0
        while not terminated:
            action = env.action_space.sample()  # Random agent
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        print(f"Random agent finished in {step_count} steps.")
        print(f"Final Info: {info}")