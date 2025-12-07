import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:33:54.172531
# Source Brief: brief_02243.md
# Brief Index: 2243
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Conduct a symphony of destruction! Use musical instruments to defeat waves of enemies in this musical shoot-'em-up."
    )
    user_guide = (
        "Controls: Use ←/↓ and ↑/→ arrow keys to aim your baton. Press space to fire musical notes and shift to cycle through unlocked instruments."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.MAX_PROJECTILES = 50

        # Action and Observation Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_instrument = pygame.font.SysFont("Consolas", 16)
        
        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_STAGE = (30, 20, 50)
        self.COLOR_PLAYER = (0, 255, 255) # Cyan
        self.COLOR_ENEMY = (255, 0, 128) # Magenta
        self.COLOR_HEALTH_FULL = (0, 255, 100)
        self.COLOR_HEALTH_EMPTY = (255, 50, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)

        # Instrument Definitions
        self.INSTRUMENTS = [
            {"name": "Flute", "color": (100, 150, 255), "speed": 8, "damage": 5, "cooldown": 8, "size": 4, "kb": 1},
            {"name": "Trumpet", "color": (255, 200, 50), "speed": 12, "damage": 15, "cooldown": 15, "size": 6, "kb": 2},
            {"name": "Tuba", "color": (200, 100, 255), "speed": 5, "damage": 30, "cooldown": 25, "size": 10, "kb": 5},
            {"name": "Violin", "color": (150, 255, 150), "speed": 10, "damage": 8, "cooldown": 20, "size": 3, "kb": 1, "burst": 3},
        ]
        
        # Initialize state variables
        # self._initialize_state_variables() is called in reset()

    def _initialize_state_variables(self):
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_health = 100.0
        self.player_baton_angle = -math.pi / 2  # Pointing up
        self.player_fire_cooldown = 0
        self.last_shift_state = False
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.unlocked_instruments = []
        self.current_instrument_idx = 0
        self.enemies_defeated_count = 0
        self.reward_this_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state_variables()

        # Spawn enemies
        num_enemies = 3
        for i in range(num_enemies):
            x_pos = self.WIDTH * (i + 1) / (num_enemies + 1)
            self.enemies.append({
                "pos": pygame.math.Vector2(x_pos, 50),
                "health": 100.0,
                "cooldown": self.np_random.integers(30, 90),
                "max_health": 100.0
            })
            
        self.unlocked_instruments = [0] # Start with Flute
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.game_over = self._check_termination()
        if self.game_over:
            return self._get_observation(), self.reward_this_step, self.game_over, False, self._get_info()

        self._handle_input(action)
        self._update_game_logic()
        self._handle_collisions()
        
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # Just terminated
            if self.player_health <= 0:
                self.reward_this_step += -100.0 # Player defeat
            elif not self.enemies:
                self.reward_this_step += 100.0 # Victory
        
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Baton rotation
        rotation_speed = 0.1
        if movement in [1, 4]: # Up/Right -> Counter-Clockwise
            self.player_baton_angle -= rotation_speed
        if movement in [2, 3]: # Down/Left -> Clockwise
            self.player_baton_angle += rotation_speed
            
        # Firing
        if space_held and self.player_fire_cooldown <= 0:
            self._fire_player_projectile()

        # Instrument cycling
        if shift_held and not self.last_shift_state:
            self.current_instrument_idx = (self.current_instrument_idx + 1) % len(self.unlocked_instruments)
        self.last_shift_state = shift_held

    def _fire_player_projectile(self):
        instrument_id = self.unlocked_instruments[self.current_instrument_idx]
        instrument = self.INSTRUMENTS[instrument_id]
        self.player_fire_cooldown = instrument["cooldown"]

        if len(self.projectiles) >= self.MAX_PROJECTILES:
            return

        vel = pygame.math.Vector2(math.cos(self.player_baton_angle), math.sin(self.player_baton_angle)) * instrument["speed"]
        
        if instrument.get("burst", 1) > 1:
            for i in range(instrument["burst"]):
                if len(self.projectiles) >= self.MAX_PROJECTILES: break
                spread_angle = (i - (instrument["burst"] - 1) / 2.0) * 0.15 # Small spread
                burst_vel = vel.rotate_rad(spread_angle)
                self.projectiles.append({
                    "pos": pygame.math.Vector2(self.player_pos), "vel": burst_vel, "owner": "player", 
                    "damage": instrument["damage"], "color": instrument["color"], 
                    "size": instrument["size"], "kb": instrument["kb"]
                })
        else:
            self.projectiles.append({
                "pos": pygame.math.Vector2(self.player_pos), "vel": vel, "owner": "player", 
                "damage": instrument["damage"], "color": instrument["color"], 
                "size": instrument["size"], "kb": instrument["kb"]
            })


    def _update_game_logic(self):
        # Update player cooldown
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
            
        # Update enemies
        difficulty_factor = 1.0 + (self.steps / self.FPS) * 0.01
        for enemy in self.enemies:
            enemy["cooldown"] -= 1
            if enemy["cooldown"] <= 0:
                if len(self.projectiles) < self.MAX_PROJECTILES:
                    angle_to_player = math.atan2(self.player_pos.y - enemy["pos"].y, self.player_pos.x - enemy["pos"].x)
                    angle_to_player += self.np_random.uniform(-0.2, 0.2) # Add some inaccuracy
                    vel = pygame.math.Vector2(math.cos(angle_to_player), math.sin(angle_to_player)) * 6
                    self.projectiles.append({
                        "pos": pygame.math.Vector2(enemy["pos"]), "vel": vel, "owner": "enemy",
                        "damage": 10, "color": self.COLOR_ENEMY, "size": 5, "kb": 0
                    })
                enemy["cooldown"] = max(20, 120 / difficulty_factor)
                
        # Update projectiles
        for p in self.projectiles:
            p["pos"] += p["vel"]
            
        # Update particles
        for particle in self.particles:
            particle["pos"] += particle["vel"]
            particle["lifespan"] -= 1
            particle["vel"] *= 0.98 # Friction

        # Cleanup dead entities
        self.projectiles = [p for p in self.projectiles if 0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT]
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _handle_collisions(self):
        # Player projectiles vs enemies
        for p in list(self.projectiles):
            if p["owner"] != "player": continue
            for enemy in list(self.enemies):
                if p["pos"].distance_to(enemy["pos"]) < 20 + p["size"]:
                    enemy["health"] -= p["damage"]
                    self.reward_this_step += 0.1
                    self.score += p["damage"]
                    self._spawn_particles(p["pos"], p["color"], 20, p["size"])
                    if p in self.projectiles: self.projectiles.remove(p)
                    
                    if enemy["health"] <= 0:
                        self.reward_this_step += 1.0
                        self.score += 100
                        self._spawn_particles(enemy["pos"], self.COLOR_ENEMY, 100, 20)
                        self.enemies.remove(enemy)
                        self.enemies_defeated_count += 1

                        # Unlock new instrument
                        if self.enemies_defeated_count % 2 == 0:
                            next_instrument = len(self.unlocked_instruments)
                            if next_instrument < len(self.INSTRUMENTS):
                                self.unlocked_instruments.append(next_instrument)
                                self.reward_this_step += 5.0
                                self.score += 500
                    break
        
        # Enemy projectiles vs player
        for p in list(self.projectiles):
            if p["owner"] != "enemy": continue
            if p["pos"].distance_to(self.player_pos) < 20 + p["size"]:
                self.player_health -= p["damage"]
                self.player_health = max(0, self.player_health)
                self.reward_this_step -= 0.1
                self._spawn_particles(p["pos"], p["color"], 20, p["size"])
                if p in self.projectiles: self.projectiles.remove(p)
        
        # Projectile vs projectile
        for i in range(len(self.projectiles)):
            for j in range(i + 1, len(self.projectiles)):
                if i >= len(self.projectiles) or j >= len(self.projectiles): continue
                p1 = self.projectiles[i]
                p2 = self.projectiles[j]
                if p1["owner"] != p2["owner"]:
                    if p1["pos"].distance_to(p2["pos"]) < p1["size"] + p2["size"]:
                        mid_point = (p1["pos"] + p2["pos"]) / 2
                        self._spawn_particles(mid_point, (255, 255, 255), 10, 5)
                        p1["pos"].y = -1000 # Mark for removal
                        p2["pos"].y = -1000 # Mark for removal
                        
    def _spawn_particles(self, pos, color, count, base_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * base_speed * 0.3
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.math.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color,
                "size": self.np_random.uniform(1, 4)
            })

    def _check_termination(self):
        if self.player_health <= 0:
            return True
        if not self.enemies:
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_STAGE, (0, 0, self.WIDTH, self.HEIGHT / 2))

        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["size"]), color)

        # Render projectiles
        for p in self.projectiles:
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), p["size"] + 3, (*p["color"], 60))
            # Note head
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), p["size"], p["color"])
            # Note stem
            stem_end = p["pos"] + pygame.math.Vector2(0, -p["size"] * 3)
            pygame.draw.line(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y)), (int(stem_end.x), int(stem_end.y)), 2)

        # Render enemies
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"].x), int(enemy["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 23, (*self.COLOR_ENEMY, 60))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 20, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 20, (255,255,255))
            self._render_health_bar(enemy["pos"] + pygame.math.Vector2(0, 30), enemy["health"], enemy["max_health"])

        # Render player
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], 23, (*self.COLOR_PLAYER, 60))
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], 20, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], 20, (255,255,255))
        
        # Render baton
        baton_end = self.player_pos + pygame.math.Vector2(math.cos(self.player_baton_angle), math.sin(self.player_baton_angle)) * 40
        pygame.draw.line(self.screen, (255, 255, 255), player_pos_int, (int(baton_end.x), int(baton_end.y)), 5)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, player_pos_int, (int(baton_end.x), int(baton_end.y)), 2)

    def _render_health_bar(self, pos, current_hp, max_hp):
        width, height = 50, 8
        x, y = pos.x - width / 2, pos.y
        
        health_ratio = max(0, current_hp / max_hp)
        
        bg_rect = pygame.Rect(int(x), int(y), width, height)
        pygame.draw.rect(self.screen, (50, 50, 50), bg_rect, border_radius=3)
        
        fg_width = int(width * health_ratio)
        fg_rect = pygame.Rect(int(x), int(y), fg_width, height)
        
        color = (
            self.COLOR_HEALTH_EMPTY[0] + (self.COLOR_HEALTH_FULL[0] - self.COLOR_HEALTH_EMPTY[0]) * health_ratio,
            self.COLOR_HEALTH_EMPTY[1] + (self.COLOR_HEALTH_FULL[1] - self.COLOR_HEALTH_EMPTY[1]) * health_ratio,
            self.COLOR_HEALTH_EMPTY[2] + (self.COLOR_HEALTH_FULL[2] - self.COLOR_HEALTH_EMPTY[2]) * health_ratio,
        )
        pygame.draw.rect(self.screen, color, fg_rect, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Player Health
        health_text = self.font_ui.render(f"Health: {int(self.player_health)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, self.HEIGHT - 30))
        self._render_health_bar(self.player_pos + pygame.math.Vector2(0, 30), self.player_health, 100)

        # Instrument Display
        start_y = self.HEIGHT - 40 - (len(self.unlocked_instruments) * 20)
        for i, instrument_id in enumerate(self.unlocked_instruments):
            instrument = self.INSTRUMENTS[instrument_id]
            is_selected = self.current_instrument_idx < len(self.unlocked_instruments) and instrument_id == self.unlocked_instruments[self.current_instrument_idx]
            
            text_color = instrument["color"] if is_selected else (100, 100, 120)
            prefix = "> " if is_selected else "  "
            
            instrument_text = self.font_instrument.render(f"{prefix}{instrument['name']}", True, text_color)
            self.screen.blit(instrument_text, (self.WIDTH - 120, start_y + i * 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "enemies_left": len(self.enemies),
            "unlocked_instruments": len(self.unlocked_instruments)
        }
        
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Unset the dummy video driver for manual play
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # Quit the headless instance
    pygame.init() # Re-init for display

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Symphony of Destruction")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        shift_pressed_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    shift_pressed_this_frame = True
        
        # Use a one-shot detection for shift to cycle instruments
        if shift_pressed_this_frame:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(env.FPS)
        
    env.close()