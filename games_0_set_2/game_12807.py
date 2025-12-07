import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:29:09.051646
# Source Brief: brief_02807.md
# Brief Index: 2807
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Cell Sentinel'.
    The player defends a central cell from invading microbes by firing projectiles
    and manipulating a magnetic field.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend a central cell from invading microbes. Rotate a launcher to fire projectiles and manipulate a magnetic field to bend their trajectory."
    )
    user_guide = (
        "Controls: ←→ to rotate launcher, ↑↓ to control magnetic field. Press space to fire and shift for a burst shot."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2500
    WIN_WAVE = 15

    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_CELL_WALL = (0, 100, 120)
    COLOR_CELL_WALL_GLOW = (0, 150, 180)
    COLOR_LAUNCHER = (100, 255, 100)
    COLOR_LAUNCHER_GLOW = (180, 255, 180)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_PROJECTILE_TRAIL = (200, 200, 50)
    COLOR_MICROBE = (255, 50, 50)
    COLOR_MICROBE_GLOW = (255, 100, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (0, 200, 0)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_MAG_FIELD_POS = (50, 50, 255, 50)  # RGBA
    COLOR_MAG_FIELD_NEG = (255, 50, 50, 50)  # RGBA

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.render_mode = render_mode
        self.cell_center = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30)
        self.cell_radius = 160

        # These attributes are reset in `reset()`
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 0
        self.current_wave = 0
        self.launcher_angle = 0
        self.magnetic_field_strength = 0.0
        self.projectiles = []
        self.microbes = []
        self.particles = []
        self.last_space_held = False
        self.fire_cooldown = 0
        self.powerup_cooldown = 0
        self.powerup_unlocked = False
        self.reward_this_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 100
        self.current_wave = 1
        self.launcher_angle = -math.pi / 2  # Pointing up
        self.magnetic_field_strength = 0.0
        self.projectiles.clear()
        self.microbes.clear()
        self.particles.clear()
        self.last_space_held = False
        self.fire_cooldown = 0
        self.powerup_cooldown = 0
        self.powerup_unlocked = False

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        self.steps += 1
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._handle_input(movement, space_held, shift_held)
        self._update_game_state()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
             if self.player_health <= 0:
                 self.reward_this_step -= 100 # Penalty for losing
             elif self.current_wave > self.WIN_WAVE:
                 self.reward_this_step += 100 # Bonus for winning
             self.game_over = True

        reward = np.clip(self.reward_this_step, -100, 100)

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Action 0 (movement)
        if movement == 1:  # Rotate CCW
            self.launcher_angle -= 0.1
        elif movement == 2:  # Rotate CW
            self.launcher_angle += 0.1
        elif movement == 3:  # Increase magnetic field
            self.magnetic_field_strength = min(1.0, self.magnetic_field_strength + 0.05)
        elif movement == 4:  # Decrease magnetic field
            self.magnetic_field_strength = max(-1.0, self.magnetic_field_strength + -0.05)
        
        self.launcher_angle %= (2 * math.pi)

        # Action 1 (space) - Fire projectile
        if space_held and not self.last_space_held and self.fire_cooldown <= 0:
            # SFX: Player_Shoot.wav
            start_pos = self.cell_center + pygame.math.Vector2(0, -self.cell_radius - 15)
            vel = pygame.math.Vector2(1, 0).rotate_rad(self.launcher_angle) * 8
            self.projectiles.append({
                "pos": start_pos, "vel": vel, "trail": deque(maxlen=10)
            })
            self.fire_cooldown = 5 # 5 frames cooldown
            self.reward_this_step -= 0.01 # Cost for firing

        # Action 2 (shift) - Activate power-up
        if self.powerup_unlocked and shift_held and self.powerup_cooldown <= 0:
            # SFX: Powerup_Activate.wav
            start_pos = self.cell_center + pygame.math.Vector2(0, -self.cell_radius - 15)
            for angle_offset in [-0.2, 0, 0.2]:
                vel = pygame.math.Vector2(1, 0).rotate_rad(self.launcher_angle + angle_offset) * 7
                self.projectiles.append({
                    "pos": start_pos, "vel": vel, "trail": deque(maxlen=10)
                })
            self.powerup_cooldown = 60 # 2 seconds cooldown
            self.reward_this_step -= 0.1 # Higher cost for powerup

        self.last_space_held = space_held

    def _update_game_state(self):
        if self.fire_cooldown > 0: self.fire_cooldown -= 1
        if self.powerup_cooldown > 0: self.powerup_cooldown -= 1
        
        # Decay magnetic field towards zero
        self.magnetic_field_strength *= 0.98

        self._update_projectiles()
        self._update_microbes()
        self._update_particles()
        self._check_wave_completion()

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["trail"].append(p["pos"].copy())
            
            # Magnetic field force: F = q(v x B) -> a = (v.y*B, -v.x*B) * const
            force = p["vel"].rotate(90) * self.magnetic_field_strength * 0.1
            p["vel"] += force
            p["pos"] += p["vel"]

            # Boundary check
            if p["pos"].distance_to(self.cell_center) > self.cell_radius:
                self.projectiles.remove(p)
                continue

            # Collision with microbes
            for m in self.microbes[:]:
                if p["pos"].distance_to(m["pos"]) < m["radius"]:
                    # SFX: Microbe_Hit.wav
                    m["health"] -= 1
                    self.reward_this_step += 0.1
                    self._create_particles(p["pos"], self.COLOR_PROJECTILE, 5)
                    if p in self.projectiles: self.projectiles.remove(p)
                    
                    if m["health"] <= 0:
                        # SFX: Microbe_Destroyed.wav
                        self.score += 10
                        self.reward_this_step += 1.0
                        self._create_particles(m["pos"], self.COLOR_MICROBE, 20)
                        self.microbes.remove(m)
                    break

    def _update_microbes(self):
        for m in self.microbes[:]:
            # Movement towards center
            direction = (self.cell_center - m["pos"]).normalize()
            speed = m["speed"]
            
            # Stun effect from strong magnetic field
            if abs(self.magnetic_field_strength) > 0.7:
                speed *= 0.3
                m["stunned"] = True
            else:
                m["stunned"] = False

            # Wobble animation
            m["wobble_angle"] += m["wobble_speed"]
            wobble_offset = pygame.math.Vector2(0, 1).rotate(m["wobble_angle"] * 57.3) * 0.2
            
            m["vel"] = direction * speed + wobble_offset
            m["pos"] += m["vel"]

            # Reached center
            if m["pos"].distance_to(self.cell_center) < 10:
                # SFX: Player_Damage.wav
                self.player_health -= 10
                self.reward_this_step -= 2.0
                self.microbes.remove(m)
                self._create_particles(self.cell_center, (255, 255, 255), 15, is_flash=True)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _spawn_wave(self):
        if self.current_wave > 5 and not self.powerup_unlocked:
            self.powerup_unlocked = True
            # SFX: Powerup_Unlocked.wav

        num_microbes = 3 + self.current_wave
        for _ in range(num_microbes):
            angle = random.uniform(0, 2 * math.pi)
            pos = self.cell_center + pygame.math.Vector2(self.cell_radius, 0).rotate_rad(angle)
            
            base_speed = 1.0
            base_health = 1
            
            speed_multiplier = 1 + (0.05 * self.current_wave)
            health_multiplier = 1 + (0.05 * self.current_wave)
            
            self.microbes.append({
                "pos": pos,
                "vel": pygame.math.Vector2(0, 0),
                "health": round(base_health * health_multiplier),
                "max_health": round(base_health * health_multiplier),
                "speed": base_speed * speed_multiplier,
                "radius": 10,
                "wobble_angle": random.uniform(0, 2 * math.pi),
                "wobble_speed": random.uniform(0.1, 0.3),
                "stunned": False
            })

    def _check_wave_completion(self):
        if not self.microbes and not self.game_over:
            self.current_wave += 1
            self.score += 50
            self.reward_this_step += 5.0
            self._spawn_wave()

    def _check_termination(self):
        return self.player_health <= 0 or self.current_wave > self.WIN_WAVE

    def _create_particles(self, pos, color, count, is_flash=False):
        for _ in range(count):
            if is_flash:
                vel = pygame.math.Vector2(0,0) # Flash particles don't move
                lifespan = 5
            else:
                vel = pygame.math.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
                lifespan = random.randint(15, 30)
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifespan": lifespan, "color": color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_microbes()
        self._render_projectiles()
        self._render_particles()
        self._render_launcher()

    def _render_background(self):
        # Cell wall
        pygame.gfxdraw.aacircle(self.screen, int(self.cell_center.x), int(self.cell_center.y), self.cell_radius, self.COLOR_CELL_WALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(self.cell_center.x), int(self.cell_center.y), self.cell_radius, self.COLOR_CELL_WALL)
        
        # Magnetic field visualization
        if abs(self.magnetic_field_strength) > 0.05:
            num_lines = int(abs(self.magnetic_field_strength) * 10)
            color = self.COLOR_MAG_FIELD_POS if self.magnetic_field_strength > 0 else self.COLOR_MAG_FIELD_NEG
            
            for i in range(num_lines):
                alpha_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                radius = self.cell_radius * (1 - i / (num_lines + 1))
                pulse = (math.sin(self.steps * 0.2 + i) + 1) / 2
                current_color = (*color[:3], int(color[3] * pulse))
                
                rect = pygame.Rect(self.cell_center.x - radius, self.cell_center.y - radius, radius * 2, radius * 2)
                
                start_angle = math.pi if self.magnetic_field_strength > 0 else 0
                stop_angle = 2 * math.pi if self.magnetic_field_strength > 0 else math.pi
                
                pygame.draw.arc(alpha_surf, current_color, rect, start_angle, stop_angle, width=3)
                self.screen.blit(alpha_surf, (0, 0))

    def _render_launcher(self):
        base_pos = self.cell_center + pygame.math.Vector2(0, -self.cell_radius - 15)
        
        # Simple triangle for launcher
        p1 = base_pos + pygame.math.Vector2(15, 0).rotate_rad(self.launcher_angle)
        p2 = base_pos + pygame.math.Vector2(-7, -7).rotate_rad(self.launcher_angle)
        p3 = base_pos + pygame.math.Vector2(-7, 7).rotate_rad(self.launcher_angle)
        
        # Glow effect
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_LAUNCHER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_LAUNCHER)

    def _render_projectiles(self):
        for p in self.projectiles:
            # Trail
            if len(p["trail"]) > 1:
                for i in range(len(p["trail"]) - 1):
                    alpha = (i / len(p["trail"])) * 255
                    pygame.draw.line(self.screen, (*self.COLOR_PROJECTILE_TRAIL, int(alpha)), p["trail"][i], p["trail"][i+1], 2)
            # Projectile head
            pos = (int(p["pos"].x), int(p["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE)

    def _render_microbes(self):
        for m in self.microbes:
            pos = (int(m["pos"].x), int(m["pos"].y))
            pulse_radius = m["radius"] * (1 + 0.1 * math.sin(m["wobble_angle"]))
            
            # Glow
            glow_color = self.COLOR_MICROBE_GLOW if not m["stunned"] else (100, 100, 255)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(pulse_radius) + 2, glow_color)
            
            # Body
            body_color = self.COLOR_MICROBE if not m["stunned"] else (150, 150, 255)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(pulse_radius), body_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(pulse_radius), (0,0,0))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, (p["lifespan"] / 30.0) * 255)
            color = (*p["color"], alpha)
            size = int(max(1, p["lifespan"] / 6))
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p["pos"].x - size), int(p["pos"].y - size)))

    def _render_ui(self):
        # Health Bar
        health_percent = max(0, self.player_health / 100.0)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_percent), 20))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Wave
        wave_text = self.font_large.render(f"WAVE {self.current_wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH/2 - wave_text.get_width()/2, 5))

        # Powerup status
        if self.powerup_unlocked:
            text = "BURST: READY" if self.powerup_cooldown <= 0 else f"BURST: {self.powerup_cooldown/self.FPS:.1f}s"
            color = self.COLOR_HEALTH_BAR if self.powerup_cooldown <= 0 else (150, 150, 150)
            powerup_text = self.font_small.render(text, True, color)
            self.screen.blit(powerup_text, (15, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "health": self.player_health
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # Example of how to run the environment
    # This part is for manual play and visualization, and is not part of the core environment
    # The environment itself runs headlessly as required
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Use Arrow Keys to rotate/control field, Space to fire, Left Shift for powerup
    obs, info = env.reset()
    done = False
    
    # Use a different screen for display to avoid conflicts with get_observation
    pygame.display.init()
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cell Sentinel")
    
    running = True
    while running:
        movement_action = 0 # no-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement_action = 1 # Rotate CCW
        elif keys[pygame.K_RIGHT]:
            movement_action = 2 # Rotate CW
        elif keys[pygame.K_UP]:
            movement_action = 3 # Increase field
        elif keys[pygame.K_DOWN]:
            movement_action = 4 # Decrease field

        if keys[pygame.K_SPACE]:
            space_action = 1

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(GameEnv.FPS)

    env.close()