import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:59:51.883069
# Source Brief: brief_01323.md
# Brief Index: 1323
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player defends a sonic base
    from invading rhythmic enemies by launching and amplifying bouncing sound waves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your sonic base from invading rhythmic enemies by launching and amplifying bouncing sound waves."
    )
    user_guide = (
        "Controls: Use ↑ and ↓ arrow keys to aim the launcher. Hold space to amplify your waves for more power."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30  # Assumed FPS for smooth interpolation

    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_AMP = (255, 255, 0)
    COLOR_WAVE = (0, 200, 255)
    COLOR_WAVE_AMP = (255, 255, 100)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_PARTICLE_HIT = (255, 150, 0)
    COLOR_PARTICLE_DESTROY = (255, 255, 100)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH = (50, 200, 50)
    COLOR_ENERGY = (0, 150, 255)
    COLOR_BAR_BG = (50, 50, 80)

    # Game Parameters
    MAX_STEPS = 1000
    MAX_BASE_HEALTH = 100
    MAX_ENERGY = 100
    ENERGY_REGEN_RATE = 0.25
    WAVE_COST = 15
    AMP_WAVE_COST = 30
    LAUNCH_COOLDOWN_FRAMES = 15 # Fires a wave every 0.5 seconds
    LAUNCHER_ANGLE_SPEED = 0.05  # Radians per step
    WAVE_SPEED = 6
    WAVE_RADIUS = 8
    WAVE_BOUNCES = 4
    ENEMY_SPAWN_RATE_INITIAL = 60 # frames
    ENEMY_SPAWN_RATE_MIN = 20

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.energy = 0
        self.launcher_pos = (0,0)
        self.launcher_angle = 0.0
        self.launcher_length = 0
        self.launch_cooldown = 0
        self.is_amplified = False
        self.waves = []
        self.enemies = []
        self.particles = []
        self.enemy_spawn_timer = 0
        self.current_enemy_spawn_rate = 0
        self.base_enemy_speed = 0
        self.base_enemy_health = 0
        
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.MAX_BASE_HEALTH
        self.energy = self.MAX_ENERGY
        
        self.launcher_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT)
        self.launcher_angle = -math.pi / 2
        self.launcher_length = 30
        self.launch_cooldown = 0
        self.is_amplified = False
        
        self.waves = []
        self.enemies = []
        self.particles = []

        self.enemy_spawn_timer = 0
        self.current_enemy_spawn_rate = self.ENEMY_SPAWN_RATE_INITIAL
        self.base_enemy_speed = 1.5
        self.base_enemy_health = 1

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Actions ---
        self._handle_actions(action)

        # --- Update Game State ---
        self._update_launcher()
        self._update_waves()
        reward += self._update_enemies()
        self._update_particles()
        self._spawn_enemies()
        
        # --- Progression ---
        if self.steps > 0 and self.steps % 100 == 0:
            self.base_enemy_speed = min(4.0, self.base_enemy_speed + 0.1)
        if self.steps > 0 and self.steps % 500 == 0:
            self.base_enemy_health += 1

        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and self.base_health <= 0:
            reward = -100 # Loss penalty
        elif truncated:
            reward = 100 # Win bonus
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, _ = action
        
        # Action 0: Movement (launcher angle)
        if movement == 1:  # Up
            self.launcher_angle -= self.LAUNCHER_ANGLE_SPEED
        elif movement == 2:  # Down
            self.launcher_angle += self.LAUNCHER_ANGLE_SPEED
        # Clamp angle to prevent shooting backwards
        self.launcher_angle = max(-math.pi + 0.1, min(-0.1, self.launcher_angle))

        # Action 1: Space (amplify)
        self.is_amplified = (space_held == 1)

    def _update_launcher(self):
        # Regenerate energy
        self.energy = min(self.MAX_ENERGY, self.energy + self.ENERGY_REGEN_RATE)
        
        # Automatic launch cooldown
        if self.launch_cooldown > 0:
            self.launch_cooldown -= 1
        else:
            cost = self.AMP_WAVE_COST if self.is_amplified else self.WAVE_COST
            if self.energy >= cost:
                self.energy -= cost
                self._launch_wave()
                self.launch_cooldown = self.LAUNCH_COOLDOWN_FRAMES
                # Sound: "pew.wav" or "PEW_AMP.wav"

    def _launch_wave(self):
        start_pos = (
            self.launcher_pos[0] + self.launcher_length * math.cos(self.launcher_angle),
            self.launcher_pos[1] + self.launcher_length * math.sin(self.launcher_angle)
        )
        velocity = (
            self.WAVE_SPEED * math.cos(self.launcher_angle),
            self.WAVE_SPEED * math.sin(self.launcher_angle)
        )
        
        wave = {
            "pos": list(start_pos),
            "vel": list(velocity),
            "radius": self.WAVE_RADIUS,
            "power": 2 if self.is_amplified else 1,
            "bounces_left": self.WAVE_BOUNCES,
            "color": self.COLOR_WAVE_AMP if self.is_amplified else self.COLOR_WAVE,
            "pulse": 0
        }
        self.waves.append(wave)

    def _update_waves(self):
        for wave in self.waves[:]:
            wave["pos"][0] += wave["vel"][0]
            wave["pos"][1] += wave["vel"][1]
            wave["pulse"] = (wave["pulse"] + 1) % self.FPS

            # Wall bouncing
            bounced = False
            if wave["pos"][0] - wave["radius"] < 0 or wave["pos"][0] + wave["radius"] > self.SCREEN_WIDTH:
                wave["vel"][0] *= -1
                wave["pos"][0] = max(wave["radius"], min(self.SCREEN_WIDTH - wave["radius"], wave["pos"][0]))
                bounced = True
            if wave["pos"][1] - wave["radius"] < 0:
                wave["vel"][1] *= -1
                wave["pos"][1] = max(wave["radius"], wave["pos"][1])
                bounced = True
            
            if bounced:
                wave["bounces_left"] -= 1
                # Sound: "bounce.wav"

            if wave["bounces_left"] <= 0 or wave["pos"][1] > self.SCREEN_HEIGHT:
                self.waves.remove(wave)

    def _spawn_enemies(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            num_enemies = self.np_random.integers(2, 5)
            pattern = self.np_random.choice(['line', 'v_shape', 'sine'])
            start_x = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
            
            for i in range(num_enemies):
                enemy = {
                    "pos": [0,0], "size": 12, "health": self.base_enemy_health, "max_health": self.base_enemy_health,
                    "pattern": pattern, "pattern_offset": i * 30, "start_x": start_x,
                    "pattern_time": 0, "speed": self.base_enemy_speed
                }
                self.enemies.append(enemy)

            self.enemy_spawn_timer = self.np_random.integers(self.ENEMY_SPAWN_RATE_MIN, int(self.current_enemy_spawn_rate))
            self.current_enemy_spawn_rate = max(self.ENEMY_SPAWN_RATE_MIN, self.current_enemy_spawn_rate * 0.99)


    def _update_enemies(self):
        step_reward = 0
        for enemy in self.enemies[:]:
            # Update position based on pattern
            enemy["pattern_time"] += 1
            t = enemy["pattern_time"]
            offset = enemy["pattern_offset"]
            
            if enemy["pattern"] == 'line':
                enemy["pos"] = [enemy["start_x"], (t + offset) * 0.5 * enemy["speed"]]
            elif enemy["pattern"] == 'v_shape':
                enemy["pos"] = [enemy["start_x"] + (offset - 60) * 1.5, t * enemy["speed"]]
            elif enemy["pattern"] == 'sine':
                enemy["pos"] = [enemy["start_x"] + math.sin((t + offset) * 0.05) * 100, t * 0.75 * enemy["speed"]]

            # Check for collision with base
            if enemy["pos"][1] + enemy["size"] > self.SCREEN_HEIGHT - 20:
                self.base_health -= 10
                self._create_explosion(enemy["pos"], 20, self.COLOR_ENEMY)
                self.enemies.remove(enemy)
                # Sound: "base_hit.wav"
                continue

            # Check for collision with waves
            for wave in self.waves[:]:
                dist_sq = (enemy["pos"][0] - wave["pos"][0])**2 + (enemy["pos"][1] - wave["pos"][1])**2
                if dist_sq < (enemy["size"] + wave["radius"])**2:
                    enemy["health"] -= wave["power"]
                    self._create_explosion(wave["pos"], 10, self.COLOR_PARTICLE_HIT)
                    step_reward += 0.1 # Reward for hitting
                    # Sound: "hit.wav"
                    try:
                        self.waves.remove(wave)
                    except ValueError:
                        pass # Wave might be removed twice if it hits two enemies at once
                    
                    if enemy["health"] <= 0:
                        self.score += 1
                        step_reward += 1.0 # Reward for destroying
                        self._create_explosion(enemy["pos"], 40, self.COLOR_PARTICLE_DESTROY)
                        self.enemies.remove(enemy)
                        # Sound: "destroy.wav"
                        break
        return step_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["vel"][1] += 0.1 # Gravity
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(15, 30),
                "color": color,
                "size": self.np_random.uniform(1, 4)
            })

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.base_health, "energy": self.energy}

    def _render_game(self):
        # Render Base Platform
        base_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 20, self.SCREEN_WIDTH, 20)
        pygame.draw.rect(self.screen, (50, 40, 90), base_rect)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (0, self.SCREEN_HEIGHT - 20), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 20), 2)

        # Render Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 30.0))))
            color_with_alpha = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["size"]), color_with_alpha)

        # Render Launcher
        launcher_color = self.COLOR_PLAYER_AMP if self.is_amplified and self.energy >= self.AMP_WAVE_COST else self.COLOR_PLAYER
        end_pos = (
            self.launcher_pos[0] + self.launcher_length * math.cos(self.launcher_angle),
            self.launcher_pos[1] + self.launcher_length * math.sin(self.launcher_angle)
        )
        pygame.draw.line(self.screen, launcher_color, self.launcher_pos, end_pos, 6)
        pygame.gfxdraw.filled_circle(self.screen, int(end_pos[0]), int(end_pos[1]), 5, launcher_color)
        pygame.gfxdraw.aacircle(self.screen, int(end_pos[0]), int(end_pos[1]), 5, launcher_color)
        
        # Cooldown indicator
        cooldown_prog = self.launch_cooldown / self.LAUNCH_COOLDOWN_FRAMES
        if cooldown_prog > 0:
            pygame.draw.arc(self.screen, (255,255,255,100), (self.launcher_pos[0]-15, self.launcher_pos[1]-15, 30, 30), 0, 2*math.pi*cooldown_prog, 2)


        # Render Waves
        for wave in self.waves:
            pos = (int(wave["pos"][0]), int(wave["pos"][1]))
            radius = int(wave["radius"])
            pulse_effect = abs(math.sin(wave["pulse"] * 0.2)) * 3
            
            # Glow effect
            for i in range(3):
                alpha = 80 - i * 25
                glow_radius = radius + pulse_effect + i * 2
                glow_color = wave["color"] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(glow_radius), glow_color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(glow_radius), glow_color)

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, wave["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, wave["color"])

        # Render Enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            size = int(enemy["size"])
            
            # Glow effect
            for i in range(3):
                alpha = 100 - i * 30
                glow_color = self.COLOR_ENEMY + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + i * 2, glow_color)
            
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, (255,200,200))
            
            # Health bar
            if enemy["health"] < enemy["max_health"]:
                health_ratio = enemy["health"] / enemy["max_health"]
                bar_width = size * 2
                pygame.draw.rect(self.screen, (100,0,0), (pos[0] - bar_width/2, pos[1] - size - 8, bar_width, 5))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH, (pos[0] - bar_width/2, pos[1] - size - 8, bar_width * health_ratio, 5))


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Steps
        steps_text = self.font_small.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 40))

        # Health Bar
        health_ratio = max(0, self.base_health / self.MAX_BASE_HEALTH)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, bar_width * health_ratio, bar_height))
        health_text = self.font_small.render("HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Energy Bar
        energy_ratio = max(0, self.energy / self.MAX_ENERGY)
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 40, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (10, 40, bar_width * energy_ratio, bar_height))
        energy_text = self.font_small.render("ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (15, 42))

    def close(self):
        pygame.quit()


# --- Manual Play / Testing Block ---
if __name__ == "__main__":
    # This block is for manual play and will not run in a headless environment
    # due to the SDL_VIDEODRIVER="dummy" setting.
    # To run this, you might need to comment out the os.environ line at the top.
    
    # Un-set the dummy driver for local rendering
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Sonic Defender")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    
    while running:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame. We just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()