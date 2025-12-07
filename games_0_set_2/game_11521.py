import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:09:54.059133
# Source Brief: brief_01521.md
# Brief Index: 1521
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player, a glowing orb, navigates a subatomic world.
    The player flips gravity to move, avoids hostile "boson" particles, and can trigger
    chain reactions in "unstable nuclei" to clear paths or destroy bosons. The goal is
    to survive for as long as possible against escalating waves of enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a subatomic world as a glowing orb, flipping gravity to avoid hostile particles and triggering chain reactions to survive."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to change the direction of gravity. Press space to trigger nearby nuclei and shift to toggle magnetic fields."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2500
    WAVE_DURATION = 600  # Steps per wave

    # Colors
    COLOR_BG = (15, 10, 30)
    COLOR_GRID = (30, 20, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_BOSON = (255, 50, 100)
    COLOR_BOSON_GLOW = (200, 20, 60)
    COLOR_NUCLEUS = (100, 100, 120)
    COLOR_EXPLOSION = (255, 200, 50)
    COLOR_EMITTER_OFF = (60, 60, 60)
    COLOR_EMITTER_ON_REPEL = (50, 255, 50)
    COLOR_EMITTER_ON_ATTRACT = (200, 50, 255)
    COLOR_TEXT = (220, 220, 240)

    # Physics & Gameplay
    GRAVITY_STRENGTH = 0.4
    PLAYER_FRICTION = 0.98
    PLAYER_MAX_SPEED = 8.0
    PLAYER_RADIUS = 10
    BOSON_RADIUS = 12
    NUCLEUS_RADIUS = 15
    EMITTER_SIZE = 20
    INITIAL_BOSONS = 3
    BOSON_BASE_SPEED = 1.0
    BOSON_PLAYER_ATTRACTION = 0.03
    MAGNETIC_FORCE_STRENGTH = 0.8
    CHAIN_REACTION_RANGE = 70
    CHAIN_REACTION_TRIGGER_DIST = 35
    PROXIMITY_PENALTY_DIST = 100

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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 16)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.gravity = None
        self.bosons = None
        self.nuclei = None
        self.emitters = None
        self.explosions = None
        self.particles = None
        self.steps = None
        self.score = None
        self.wave = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        
        self.reset()
        
        # This is a critical self-check
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.gravity = pygame.math.Vector2(0, self.GRAVITY_STRENGTH)  # Start with downward gravity

        self._generate_level()

        self.explosions = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Actions ---
        reward += self._handle_input(movement, space_held, shift_held)

        # --- Update Game State ---
        self._update_player()
        self._update_bosons()
        reward += self._update_explosions()
        self._update_particles()
        
        # --- Calculate Rewards & Check Termination ---
        proximity_penalty, collision = self._check_player_collisions()
        reward -= proximity_penalty
        
        if collision:
            self.game_over = True
            reward = -100.0 # Terminal penalty
        else:
            reward += 0.1 # Survival reward
        
        self.score += reward
        self.steps += 1
        
        # --- Wave Progression ---
        if not self.game_over and self.steps > 0 and self.steps % self.WAVE_DURATION == 0:
            self._next_wave()
            reward += 10.0

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        # Update previous action states for rising edge detection
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _generate_level(self):
        # Place unstable nuclei
        self.nuclei = []
        for _ in range(15):
            pos = pygame.math.Vector2(
                random.uniform(50, self.SCREEN_WIDTH - 50),
                random.uniform(50, self.SCREEN_HEIGHT - 50)
            )
            # Ensure they don't spawn on top of each other
            if not any(p.distance_to(pos) < self.NUCLEUS_RADIUS * 3 for p in self.nuclei):
                 self.nuclei.append(pos)

        # Place magnetic field emitters
        self.emitters = []
        emitter_positions = [
            (100, 100), (self.SCREEN_WIDTH - 100, 100),
            (100, self.SCREEN_HEIGHT - 100), (self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT - 100),
            (self.SCREEN_WIDTH / 2, 50), (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        ]
        for i, pos in enumerate(emitter_positions):
            self.emitters.append({
                "pos": pygame.math.Vector2(pos),
                "active": False,
                "type": "repel" if i % 2 == 0 else "attract" # Alternate repel/attract
            })
        
        # Spawn initial bosons
        self.bosons = []
        self._spawn_bosons(self.INITIAL_BOSONS)
        
    def _spawn_bosons(self, count):
        for _ in range(count):
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                pos = pygame.math.Vector2(random.uniform(0, self.SCREEN_WIDTH), -self.BOSON_RADIUS)
            elif edge == 'bottom':
                pos = pygame.math.Vector2(random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.BOSON_RADIUS)
            elif edge == 'left':
                pos = pygame.math.Vector2(-self.BOSON_RADIUS, random.uniform(0, self.SCREEN_HEIGHT))
            else: # right
                pos = pygame.math.Vector2(self.SCREEN_WIDTH + self.BOSON_RADIUS, random.uniform(0, self.SCREEN_HEIGHT))
            
            self.bosons.append({"pos": pos, "vel": pygame.math.Vector2(0, 0)})

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # Movement changes gravity direction
        if movement == 1: self.gravity = pygame.math.Vector2(0, -self.GRAVITY_STRENGTH)
        elif movement == 2: self.gravity = pygame.math.Vector2(0, self.GRAVITY_STRENGTH)
        elif movement == 3: self.gravity = pygame.math.Vector2(-self.GRAVITY_STRENGTH, 0)
        elif movement == 4: self.gravity = pygame.math.Vector2(self.GRAVITY_STRENGTH, 0)

        # Space action (rising edge) - trigger chain reaction
        if space_held and not self.prev_space_held and self.nuclei:
            # Find closest nucleus
            closest_n, closest_dist = None, float('inf')
            for i, nucleus_pos in enumerate(self.nuclei):
                dist = self.player_pos.distance_to(nucleus_pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_n = i
            
            if closest_n is not None and closest_dist < self.CHAIN_REACTION_TRIGGER_DIST:
                pos = self.nuclei.pop(closest_n)
                self.explosions.append({"pos": pos, "radius": 0, "max_radius": self.CHAIN_REACTION_RANGE, "speed": 4})
                self._create_particles(pos, self.COLOR_EXPLOSION, 30)
                reward += 1.0 # Reward for starting a reaction
                # SFX: Chain reaction start

        # Shift action (rising edge) - toggle magnetic field
        if shift_held and not self.prev_shift_held and self.emitters:
            closest_e = min(self.emitters, key=lambda e: self.player_pos.distance_to(e["pos"]))
            closest_e["active"] = not closest_e["active"]
            # SFX: Emitter toggle
            
        return reward

    def _update_player(self):
        self.player_vel += self.gravity
        self.player_vel *= self.PLAYER_FRICTION
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        self.player_pos += self.player_vel

        # Boundary checks
        if self.player_pos.x < self.PLAYER_RADIUS:
            self.player_pos.x = self.PLAYER_RADIUS
            self.player_vel.x *= -0.5
        if self.player_pos.x > self.SCREEN_WIDTH - self.PLAYER_RADIUS:
            self.player_pos.x = self.SCREEN_WIDTH - self.PLAYER_RADIUS
            self.player_vel.x *= -0.5
        if self.player_pos.y < self.PLAYER_RADIUS:
            self.player_pos.y = self.PLAYER_RADIUS
            self.player_vel.y *= -0.5
        if self.player_pos.y > self.SCREEN_HEIGHT - self.PLAYER_RADIUS:
            self.player_pos.y = self.SCREEN_HEIGHT - self.PLAYER_RADIUS
            self.player_vel.y *= -0.5
    
    def _update_bosons(self):
        boson_speed_multiplier = self.BOSON_BASE_SPEED + (self.wave - 1) * 0.1
        for boson in self.bosons:
            # Move towards player
            to_player = self.player_pos - boson["pos"]
            if to_player.length() > 0:
                boson["vel"] += to_player.normalize() * self.BOSON_PLAYER_ATTRACTION
            
            # Apply magnetic fields
            for emitter in self.emitters:
                if emitter["active"]:
                    to_emitter = emitter["pos"] - boson["pos"]
                    dist_sq = to_emitter.length_squared()
                    if dist_sq > 1:
                        force_dir = -to_emitter.normalize() if emitter["type"] == "repel" else to_emitter.normalize()
                        # Force falls off with distance squared
                        force_magnitude = self.MAGNETIC_FORCE_STRENGTH * 5000 / dist_sq
                        boson["vel"] += force_dir * force_magnitude

            # Apply friction and speed limit
            boson["vel"] *= 0.99
            if boson["vel"].length() > boson_speed_multiplier:
                boson["vel"].scale_to_length(boson_speed_multiplier)
            
            boson["pos"] += boson["vel"]

            # Boundary bounce for bosons
            if boson["pos"].x < self.BOSON_RADIUS or boson["pos"].x > self.SCREEN_WIDTH - self.BOSON_RADIUS:
                boson["vel"].x *= -1
            if boson["pos"].y < self.BOSON_RADIUS or boson["pos"].y > self.SCREEN_HEIGHT - self.BOSON_RADIUS:
                boson["vel"].y *= -1
    
    def _update_explosions(self):
        reward = 0
        newly_triggered_nuclei_indices = set()

        for explosion in self.explosions:
            explosion["radius"] += explosion["speed"]
            
            # Check for chain reactions with other nuclei
            for i in range(len(self.nuclei) - 1, -1, -1):
                if i not in newly_triggered_nuclei_indices:
                    if self.nuclei[i].distance_to(explosion["pos"]) < explosion["radius"]:
                        newly_triggered_nuclei_indices.add(i)

            # Check for collisions with bosons
            for i in range(len(self.bosons) - 1, -1, -1):
                if self.bosons[i]["pos"].distance_to(explosion["pos"]) < explosion["radius"] + self.BOSON_RADIUS:
                    self._create_particles(self.bosons[i]["pos"], self.COLOR_BOSON, 20)
                    self.bosons.pop(i)
                    reward += 5.0 # Reward for destroying a boson
                    # SFX: Boson destroyed
        
        # Trigger new explosions from the set of hit nuclei
        if newly_triggered_nuclei_indices:
            # Sort indices in reverse to avoid messing up list order during pop
            for i in sorted(list(newly_triggered_nuclei_indices), reverse=True):
                pos = self.nuclei.pop(i)
                self.explosions.append({"pos": pos, "radius": 0, "max_radius": self.CHAIN_REACTION_RANGE, "speed": 4})
                self._create_particles(pos, self.COLOR_EXPLOSION, 30)
                reward += 1.0 # Reward for continuing a chain
        
        # Remove finished explosions
        self.explosions = [e for e in self.explosions if e["radius"] < e["max_radius"]]
        return reward

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]
        
    def _check_player_collisions(self):
        proximity_penalty = 0
        for boson in self.bosons:
            dist = self.player_pos.distance_to(boson["pos"])
            # Player-Boson collision
            if dist < self.PLAYER_RADIUS + self.BOSON_RADIUS:
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 50)
                # SFX: Player death
                return 0, True # Collision detected
            # Proximity penalty
            if dist < self.PROXIMITY_PENALTY_DIST:
                proximity_penalty += 0.5 * (1 - dist / self.PROXIMITY_PENALTY_DIST)
        return proximity_penalty, False
        
    def _next_wave(self):
        self.wave += 1
        self._spawn_bosons(1) # Add one new boson per wave
        # SFX: Wave complete
        
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({"pos": pygame.math.Vector2(pos), "vel": vel, "life": random.randint(20, 40), "color": color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave}

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw magnetic fields
        for emitter in self.emitters:
            if emitter["active"]:
                color = self.COLOR_EMITTER_ON_REPEL if emitter["type"] == "repel" else self.COLOR_EMITTER_ON_ATTRACT
                self._draw_glow_circle(self.screen, color, emitter["pos"], self.EMITTER_SIZE // 2, 2, max_alpha=60)
        
        # Draw emitters
        for emitter in self.emitters:
            color = self.COLOR_EMITTER_OFF
            if emitter["active"]:
                color = self.COLOR_EMITTER_ON_REPEL if emitter["type"] == "repel" else self.COLOR_EMITTER_ON_ATTRACT
            rect = pygame.Rect(emitter["pos"].x - self.EMITTER_SIZE / 2, emitter["pos"].y - self.EMITTER_SIZE / 2, self.EMITTER_SIZE, self.EMITTER_SIZE)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, width=2, border_radius=3)

        # Draw nuclei
        for pos in self.nuclei:
            self._draw_glow_circle(self.screen, self.COLOR_NUCLEUS, pos, self.NUCLEUS_RADIUS, 3, max_alpha=50)

        # Draw explosions
        for explosion in self.explosions:
            alpha = 255 * (1 - explosion["radius"] / explosion["max_radius"])
            if alpha > 0:
                self._draw_glow_circle(self.screen, self.COLOR_EXPLOSION, explosion["pos"], explosion["radius"], 5, max_alpha=alpha)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 40.0))
            color = (*p["color"], alpha)
            size = max(1, int(3 * (p["life"] / 40.0)))
            # Create a temporary surface for each particle to handle alpha properly
            particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color, (size, size), size)
            self.screen.blit(particle_surf, (p["pos"].x - size, p["pos"].y - size))


        # Draw bosons
        for boson in self.bosons:
            self._draw_glow_circle(self.screen, self.COLOR_BOSON_GLOW, boson["pos"], self.BOSON_RADIUS, 4, max_alpha=100)
            pygame.gfxdraw.aacircle(self.screen, int(boson["pos"].x), int(boson["pos"].y), self.BOSON_RADIUS, self.COLOR_BOSON)
            pygame.gfxdraw.filled_circle(self.screen, int(boson["pos"].x), int(boson["pos"].y), self.BOSON_RADIUS, self.COLOR_BOSON)

        # Draw player
        if not self.game_over:
            self._draw_glow_circle(self.screen, self.COLOR_PLAYER_GLOW, self.player_pos, self.PLAYER_RADIUS, 5, max_alpha=150)
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        wave_text = self.font.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            over_text = self.font.render("GAME OVER", True, self.COLOR_BOSON)
            over_rect = over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(over_text, over_rect)

    def _draw_glow_circle(self, surface, color, center, radius, num_layers, max_alpha):
        center_int = (int(center.x), int(center.y))
        for i in range(num_layers, 0, -1):
            alpha = max_alpha * (1 - (i / num_layers))**2
            glow_color = (*color, int(alpha))
            glow_radius = int(radius + i * 2)
            
            # Create a temporary surface for the glow
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            
            # Scale it down for a softer edge
            temp_surf = pygame.transform.smoothscale(temp_surf, (int(temp_surf.get_width()*0.9), int(temp_surf.get_height()*0.9)))
            
            surface.blit(temp_surf, (center_int[0] - temp_surf.get_width()//2, center_int[1] - temp_surf.get_height()//2), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up a real display for manual play
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Subatomic Survival")
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
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Waves Survived: {info['wave']}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)

    env.close()