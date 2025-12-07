import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:48:31.632252
# Source Brief: brief_01324.md
# Brief Index: 1324
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your tempo line from descending invaders by firing projectiles from a central launcher. "
        "Use resources gained from hits to build and upgrade defensive barriers."
    )
    user_guide = (
        "Controls: Use ↑/↓ to aim and ←/→ to set power. Press space to fire a projectile. "
        "Press shift to spend tempo and upgrade the nearest barrier."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 20, 30)
    COLOR_GRID = (25, 40, 55)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PROJECTILE = (0, 200, 255)
    COLOR_ENEMY = (255, 50, 100)
    COLOR_BARRIER_BASE = (50, 100, 200)
    COLOR_TEMPO = (255, 200, 0)
    COLOR_SONIC_BOOM = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (20, 20, 30)

    # Game Parameters
    MAX_STEPS = 1000
    TEMPO_LINE_Y = 350
    LAUNCHER_POS = (SCREEN_WIDTH // 2, TEMPO_LINE_Y)
    
    INITIAL_TEMPO = 100
    MAX_TEMPO = 200
    TEMPO_HIT_COST = 25
    
    BARRIER_COUNT = 5
    BARRIER_UPGRADE_COST = 15
    BARRIER_MAX_LEVEL = 3

    PROJECTILE_SPEED_MIN = 5
    PROJECTILE_SPEED_MAX = 15
    PROJECTILE_RADIUS = 5

    ENEMY_RADIUS = 8
    INITIAL_ENEMY_SPAWN_CHANCE = 0.02
    INITIAL_ENEMY_SPEED = 1.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tempo_resource = 0
        self.launch_angle = 0.0
        self.launch_power = 0.0
        self.projectiles = []
        self.enemies = []
        self.barriers = []
        self.particles = []
        self.sonic_booms = []
        self.enemy_spawn_chance = 0.0
        self.enemy_base_speed = 0.0
        self.combo_hits = 0
        self.combo_timer = 0
        self.combo_multiplier = 1
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reward_this_step = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tempo_resource = self.INITIAL_TEMPO
        
        self.launch_angle = -math.pi / 2
        self.launch_power = (self.PROJECTILE_SPEED_MIN + self.PROJECTILE_SPEED_MAX) / 2

        self.projectiles = []
        self.enemies = []
        self.particles = deque(maxlen=200) # Performance cap
        self.sonic_booms = []
        
        self.enemy_spawn_chance = self.INITIAL_ENEMY_SPAWN_CHANCE
        self.enemy_base_speed = self.INITIAL_ENEMY_SPEED

        self.combo_hits = 0
        self.combo_timer = 0
        self.combo_multiplier = 1

        self.prev_space_held = False
        self.prev_shift_held = False

        self._initialize_barriers()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.reward_this_step = 0.01  # Survival reward

        self._handle_input(movement, space_held, shift_held)
        self._update_game_state()
        
        self.steps += 1
        self.score += self.reward_this_step

        terminated = self._check_termination()
        if terminated:
            if self.steps >= self.MAX_STEPS:
                self.reward_this_step += 100 # Victory bonus
            else:
                self.reward_this_step -= 100 # Defeat penalty
            self.score += self.reward_this_step

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _initialize_barriers(self):
        self.barriers = []
        spacing = self.SCREEN_WIDTH / (self.BARRIER_COUNT + 1)
        for i in range(self.BARRIER_COUNT):
            x = spacing * (i + 1)
            y = self.TEMPO_LINE_Y - 100
            self.barriers.append({
                "pos": pygame.Vector2(x, y),
                "level": 1,
                "health": 100,
                "max_health": 100,
                "width": 50,
                "height": 10,
                "pulse": 0.0,
            })

    def _handle_input(self, movement, space_held, shift_held):
        # Adjust angle
        if movement == 1: # Up
            self.launch_angle -= 0.05
        elif movement == 2: # Down
            self.launch_angle += 0.05
        self.launch_angle = max(-math.pi * 0.9, min(-math.pi * 0.1, self.launch_angle))

        # Adjust power
        if movement == 3: # Left
            self.launch_power -= 0.2
        elif movement == 4: # Right
            self.launch_power += 0.2
        self.launch_power = max(self.PROJECTILE_SPEED_MIN, min(self.PROJECTILE_SPEED_MAX, self.launch_power))
        
        # Launch projectile on space press
        if space_held and not self.prev_space_held:
            # sfx: player_shoot.wav
            vel = pygame.Vector2(math.cos(self.launch_angle), math.sin(self.launch_angle)) * self.launch_power
            self.projectiles.append({
                "pos": pygame.Vector2(self.LAUNCHER_POS),
                "vel": vel,
                "trail": deque(maxlen=15),
                "bounces": 0
            })
            self._create_particles(self.LAUNCHER_POS, 10, self.COLOR_PLAYER, 2, 0.5)

        # Upgrade barrier on shift press
        if shift_held and not self.prev_shift_held:
            self._upgrade_closest_barrier()

    def _upgrade_closest_barrier(self):
        if not self.barriers or self.tempo_resource < self.BARRIER_UPGRADE_COST:
            # sfx: action_fail.wav
            return

        closest_barrier = min(self.barriers, key=lambda b: abs(b["pos"].x - self.LAUNCHER_POS[0]))
        
        if closest_barrier["level"] < self.BARRIER_MAX_LEVEL:
            # sfx: upgrade.wav
            self.tempo_resource -= self.BARRIER_UPGRADE_COST
            closest_barrier["level"] += 1
            closest_barrier["max_health"] = 100 * closest_barrier["level"]
            closest_barrier["health"] = closest_barrier["max_health"]
            self._create_particles(closest_barrier["pos"], 20, self.COLOR_TEMPO, 3, 1)


    def _update_game_state(self):
        # Update difficulty
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_base_speed += 0.05
        if self.steps > 0 and self.steps % 50 == 0:
            self.enemy_spawn_chance += 0.005

        # Spawn enemies
        if self.np_random.random() < self.enemy_spawn_chance:
            self._spawn_enemy()
        
        hits_this_frame = 0

        # Update projectiles
        for p in self.projectiles[:]:
            p["trail"].append(p["pos"].copy())
            p["pos"] += p["vel"]
            
            # Wall collisions
            if not (0 < p["pos"].x < self.SCREEN_WIDTH):
                p["vel"].x *= -1
                p["pos"].x = max(1, min(self.SCREEN_WIDTH - 1, p["pos"].x))
                p["bounces"] += 1
            if p["pos"].y < 0:
                p["vel"].y *= -1
                p["pos"].y = 1
                p["bounces"] += 1

            # Remove old projectiles
            if p["pos"].y > self.SCREEN_HEIGHT or p["bounces"] > 5:
                self.projectiles.remove(p)
                continue
            
            # Enemy collisions
            for e in self.enemies[:]:
                if p["pos"].distance_to(e["pos"]) < self.PROJECTILE_RADIUS + self.ENEMY_RADIUS:
                    # sfx: enemy_hit.wav
                    self.enemies.remove(e)
                    self.projectiles.remove(p)
                    self.reward_this_step += 0.1
                    self.tempo_resource = min(self.MAX_TEMPO, self.tempo_resource + 5)
                    self._create_particles(e["pos"], 15, self.COLOR_ENEMY, 4, 1.5)
                    hits_this_frame += 1
                    self.combo_hits += 1
                    self.combo_timer = 30 # frames
                    break
        
        # Sonic Boom check
        if hits_this_frame >= 3:
            # sfx: sonic_boom.wav
            self.reward_this_step += 1.0
            self.sonic_booms.append({"pos": self.LAUNCHER_POS, "radius": 0, "max_radius": self.SCREEN_WIDTH, "life": 30})

        # Update enemies
        for e in self.enemies[:]:
            e["pos"].y += self.enemy_base_speed
            
            # Barrier collision
            hit_barrier = False
            for b in self.barriers:
                barrier_rect = pygame.Rect(b["pos"].x - b["width"] / 2, b["pos"].y - b["height"] / 2, b["width"], b["height"])
                if barrier_rect.collidepoint(e["pos"]):
                    # sfx: barrier_hit.wav
                    b["health"] -= 50
                    self.enemies.remove(e)
                    self._create_particles(e["pos"], 5, self.COLOR_BARRIER_BASE, 2, 0.8)
                    if b["health"] <= 0:
                        # sfx: barrier_break.wav
                        self._create_particles(b["pos"], 30, self.COLOR_BARRIER_BASE, 5, 2)
                        b["level"] = max(1, b["level"] - 1)
                        b["max_health"] = 100 * b["level"]
                        b["health"] = b["max_health"]
                    hit_barrier = True
                    break
            if hit_barrier:
                continue

            # Tempo line collision
            if e["pos"].y > self.TEMPO_LINE_Y:
                # sfx: tempo_line_damage.wav
                self.enemies.remove(e)
                self.tempo_resource -= self.TEMPO_HIT_COST
                self.combo_multiplier = 1
                self.combo_hits = 0
                self.sonic_booms.append({"pos": (e["pos"].x, self.TEMPO_LINE_Y), "radius": 0, "max_radius": 100, "life": 15, "color": self.COLOR_ENEMY})

        # Update barriers
        for b in self.barriers:
            b["pulse"] = (b["pulse"] + 0.1) % (2 * math.pi)

        # Update combo
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            if self.combo_hits > 2:
                self.combo_multiplier = min(10, self.combo_multiplier + 1)
            else:
                self.combo_multiplier = 1
            self.combo_hits = 0
        
        # Update particles
        for p in list(self.particles):
            p["pos"] += p["vel"]
            p["vel"].y += 0.05 # gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        # Update sonic booms
        for s in self.sonic_booms[:]:
            s["radius"] += s["max_radius"] / s["life"]
            s["life"] -= 1
            if s["life"] <= 0:
                self.sonic_booms.remove(s)

    def _spawn_enemy(self):
        x = self.np_random.uniform(self.ENEMY_RADIUS, self.SCREEN_WIDTH - self.ENEMY_RADIUS)
        self.enemies.append({
            "pos": pygame.Vector2(x, -self.ENEMY_RADIUS),
        })

    def _create_particles(self, pos, count, color, speed_max, life_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_max)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": self.np_random.uniform(10, 20) * life_mult,
                "color": color
            })

    def _check_termination(self):
        if self.tempo_resource <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self._render_background()
        self._render_barriers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_sonic_booms()
        self._render_launcher()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tempo": self.tempo_resource,
            "combo": self.combo_multiplier
        }
    
    # --- Rendering Methods ---
    def _render_text(self, text, pos, font, color, shadow_color=None, center=False):
        text_surf = font.render(str(text), True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        if shadow_color:
            shadow_surf = font.render(str(text), True, shadow_color)
            shadow_rect = shadow_surf.get_rect(topleft=(text_rect.left + 2, text_rect.top + 2))
            if center:
                shadow_rect.center = (pos[0] + 2, pos[1] + 2)
            self.screen.blit(shadow_surf, shadow_rect)
            
        self.screen.blit(text_surf, text_rect)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Pulsing grid
        pulse = (math.sin(self.steps * 0.05) + 1) / 2 * 5 + 5
        grid_color = tuple(min(255, c + pulse) for c in self.COLOR_GRID)
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, grid_color, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, grid_color, (0, i), (self.SCREEN_WIDTH, i))

        # Tempo line
        tempo_pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 10
        line_color = (
            min(255, self.COLOR_TEMPO[0] * 0.5 + tempo_pulse),
            min(255, self.COLOR_TEMPO[1] * 0.5 + tempo_pulse),
            min(255, self.COLOR_TEMPO[2] * 0.5)
        )
        pygame.draw.line(self.screen, line_color, (0, self.TEMPO_LINE_Y), (self.SCREEN_WIDTH, self.TEMPO_LINE_Y), 2)

    def _render_launcher(self):
        x, y = self.LAUNCHER_POS
        # Base
        pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 12, self.COLOR_BG)
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 12, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 8, self.COLOR_PLAYER)

        # Aiming reticle
        power_ratio = (self.launch_power - self.PROJECTILE_SPEED_MIN) / (self.PROJECTILE_SPEED_MAX - self.PROJECTILE_SPEED_MIN)
        length = 20 + power_ratio * 30
        end_x = x + math.cos(self.launch_angle) * length
        end_y = y + math.sin(self.launch_angle) * length
        reticle_color = self.COLOR_PLAYER
        pygame.draw.aaline(self.screen, reticle_color, (x, y), (end_x, end_y), 2)
        pygame.gfxdraw.aacircle(self.screen, int(end_x), int(end_y), 4, reticle_color)

    def _render_projectiles(self):
        for p in self.projectiles:
            # Trail
            for i, pos in enumerate(p["trail"]):
                alpha = int(255 * (i / len(p["trail"])))
                color = (*self.COLOR_PROJECTILE, alpha)
                if alpha > 10:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(self.PROJECTILE_RADIUS * (i / len(p["trail"]))), color)

            # Main projectile
            px, py = int(p["pos"].x), int(p["pos"].y)
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.PROJECTILE_RADIUS + 3, (*self.COLOR_PROJECTILE, 60))
            # Core
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, px, py, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

    def _render_enemies(self):
        for e in self.enemies:
            ex, ey = int(e["pos"].x), int(e["pos"].y)
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, ex, ey, self.ENEMY_RADIUS + 4, (*self.COLOR_ENEMY, 80))
            # Core
            pygame.gfxdraw.filled_circle(self.screen, ex, ey, self.ENEMY_RADIUS, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, ex, ey, self.ENEMY_RADIUS, self.COLOR_ENEMY)

    def _render_barriers(self):
        for b in self.barriers:
            pulse_size = math.sin(b["pulse"]) * 2
            color_mult = (b["level"] / self.BARRIER_MAX_LEVEL) * 0.7 + 0.3
            color = tuple(min(255, c * color_mult) for c in self.COLOR_BARRIER_BASE)
            
            rect = pygame.Rect(
                b["pos"].x - (b["width"] + pulse_size) / 2,
                b["pos"].y - (b["height"] + pulse_size/2) / 2,
                b["width"] + pulse_size,
                b["height"] + pulse_size/2
            )
            # Glow
            glow_rect = rect.inflate(8, 8)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*color, 60), glow_surf.get_rect(), border_radius=5)
            self.screen.blit(glow_surf, glow_rect)
            
            # Core barrier
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            
            # Health bar
            if b["health"] < b["max_health"]:
                health_ratio = b["health"] / b["max_health"]
                health_bar_rect = pygame.Rect(rect.left, rect.top - 7, b["width"], 4)
                pygame.draw.rect(self.screen, self.COLOR_GRID, health_bar_rect, border_radius=2)
                health_bar_rect.width *= health_ratio
                pygame.draw.rect(self.screen, self.COLOR_PLAYER if health_ratio > 0.5 else self.COLOR_ENEMY, health_bar_rect, border_radius=2)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, int(255 * (p["life"] / 20)))
            size = max(1, int(3 * (p["life"] / 20)))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), size, color)

    def _render_sonic_booms(self):
        for s in self.sonic_booms:
            alpha = int(255 * (s["life"] / (s["max_radius"] / 10)))
            if alpha > 0:
                color = s.get("color", self.COLOR_SONIC_BOOM)
                pygame.gfxdraw.aacircle(self.screen, int(s["pos"][0]), int(s["pos"][1]), int(s["radius"]), (*color, alpha))
                pygame.gfxdraw.aacircle(self.screen, int(s["pos"][0]), int(s["pos"][1]), int(s["radius"] * 0.9), (*color, alpha))

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {int(self.score)}", (10, 10), self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Tempo Resource Bar
        bar_width = 200
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 15
        
        tempo_ratio = self.tempo_resource / self.MAX_TEMPO
        
        # Background bar
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_GRID, bg_rect, border_radius=5)
        
        # Fill bar
        fill_rect = pygame.Rect(bar_x, bar_y, bar_width * tempo_ratio, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_TEMPO, fill_rect, border_radius=5)
        
        self._render_text(f"TEMPO", (self.SCREEN_WIDTH / 2, bar_y + bar_height / 2), self.font_small, self.COLOR_BG, center=True)

        # Combo Multiplier
        if self.combo_multiplier > 1:
            self._render_text(f"x{self.combo_multiplier}", (self.SCREEN_WIDTH - 70, self.TEMPO_LINE_Y - 30), self.font_large, self.COLOR_TEMPO, self.COLOR_TEXT_SHADOW, center=True)
            
        # Game Over Text
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "VICTORY" if self.steps >= self.MAX_STEPS else "GAME OVER"
            self._render_text(msg, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20), self.font_large, self.COLOR_TEMPO, center=True)
            self._render_text(f"FINAL SCORE: {int(self.score)}", (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20), self.font_small, self.COLOR_TEXT, center=True)


    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For manual play, we need a real display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tempo Defender")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        # This is for manual control, not used by the agent
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    print(f"Game Over! Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
    env.close()