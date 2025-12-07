import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:06:34.858980
# Source Brief: brief_00864.md
# Brief Index: 864
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player manipulates time to survive.

    The player, a bright white circle, must survive waves of red enemy squares.
    The core mechanic is weaving "paradoxes" (blue pulsating circles) which
    slow enemies within their radius. The player can teleport to existing
    paradoxes to escape danger.

    Managing "Paradox Stability" is key. It depletes over time, when enemies
    touch paradoxes, and especially when an enemy touches the player.
    Collecting green resource orbs allows the player to weave more paradoxes.

    The game ends if Paradox Stability reaches zero or if the player survives
    for the maximum number of steps.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - action[1]: Weave Paradox (1: hold space)
    - action[2]: Teleport (1: hold shift)
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Survive waves of enemies by manipulating time. Weave paradoxes to slow foes and teleport between them to escape danger."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to weave a paradox and shift to teleport to the oldest one."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 55)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (200, 40, 40)
        self.COLOR_PARADOX = (0, 150, 255)
        self.COLOR_PARADOX_GLOW = (0, 100, 200)
        self.COLOR_RESOURCE = (50, 255, 150)
        self.COLOR_RESOURCE_GLOW = (40, 200, 120)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_STABILITY_BAR_HIGH = (0, 255, 150)
        self.COLOR_STABILITY_BAR_MID = (255, 200, 0)
        self.COLOR_STABILITY_BAR_LOW = (255, 50, 50)

        # Game Parameters
        self.PLAYER_SPEED = 5
        self.PLAYER_RADIUS = 8
        self.ENEMY_RADIUS = 10
        self.ENEMY_BASE_SPEED = 1.0
        self.RESOURCE_RADIUS = 6
        self.PARADOX_RADIUS = 80
        self.PARADOX_LIFETIME = 300  # in steps
        self.PARADOX_COST = 1
        self.PARADOX_COOLDOWN_MAX = 30
        self.TELEPORT_COOLDOWN_MAX = 60
        self.INITIAL_RESOURCES = 3
        self.INITIAL_RESOURCE_NODES = 5
        self.MAX_RESOURCE_NODES = 10
        self.STABILITY_DECAY_BASE = 0.02
        self.STABILITY_DECAY_ENEMY_CONTACT = 1.5
        self.STABILITY_DECAY_PLAYER_CONTACT = 15.0

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        # State variables
        self.steps = 0
        self.score = 0
        self.paradox_stability = 0.0
        self.resources = 0
        self.player_pos = [0, 0]
        self.enemies = []
        self.paradoxes = []
        self.resource_nodes = []
        self.particles = []
        self.paradox_cooldown = 0
        self.teleport_cooldown = 0
        self.enemy_speed_multiplier = 1.0
        self.num_enemies_to_spawn = 1
        self.just_teleported = 0 # Timer for visual effect

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.paradox_stability = 100.0
        self.resources = self.INITIAL_RESOURCES

        self.player_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        
        self.enemies = []
        self.paradoxes = []
        self.resource_nodes = []
        self.particles = []

        self.paradox_cooldown = 0
        self.teleport_cooldown = 0
        self.just_teleported = 0

        self.enemy_speed_multiplier = 1.0
        self.num_enemies_to_spawn = 1
        
        for _ in range(self.INITIAL_RESOURCE_NODES):
            self._spawn_resource()
        
        self._spawn_enemies()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        old_stability = self.paradox_stability

        # --- Update Timers & Cooldowns ---
        self._update_cooldowns()

        # --- Handle Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_movement(movement)
        if self._handle_paradox_weaving(space_held):
            # SFX: Paradox Weave sound
            pass
        teleport_reward = self._handle_teleport(shift_held)
        reward += teleport_reward

        # --- Update Game Entities ---
        self._update_enemies()
        stability_loss_from_paradox_decay = self._update_paradoxes()
        self._update_particles()

        # --- Handle Collisions & Interactions ---
        stability_loss_from_enemy_hits = self._handle_enemy_paradox_collisions()
        stability_loss_from_player_hit = self._handle_player_enemy_collisions()
        resource_reward = self._handle_player_resource_collisions()
        reward += resource_reward

        # --- Update Global State & Progression ---
        total_stability_loss = (
            self.STABILITY_DECAY_BASE +
            stability_loss_from_paradox_decay +
            stability_loss_from_enemy_hits +
            stability_loss_from_player_hit
        )
        self.paradox_stability = max(0, self.paradox_stability - total_stability_loss)
        self._update_progression()
        self._spawn_enemies()
        if len(self.resource_nodes) < self.MAX_RESOURCE_NODES and self.np_random.random() < 0.02:
            self._spawn_resource()

        # --- Calculate Final Rewards for Step ---
        stability_lost_this_step = old_stability - self.paradox_stability
        reward -= stability_lost_this_step * 0.5
        if self.paradox_stability > 50:
            reward += 0.1

        # --- Check Termination ---
        self.steps += 1
        terminated = self.paradox_stability <= 0 or self.steps >= self.MAX_STEPS
        truncated = False # This game does not truncate based on time limit
        if self.steps >= self.MAX_STEPS:
            terminated = True
            reward = 100.0
            self.score += 10000
        elif self.paradox_stability <= 0:
            terminated = True
            reward = -100.0

        # Update display score
        self.score += int(resource_reward * 10) + int(teleport_reward * 50)

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    # --- Helper Methods for step() ---

    def _update_cooldowns(self):
        self.paradox_cooldown = max(0, self.paradox_cooldown - 1)
        self.teleport_cooldown = max(0, self.teleport_cooldown - 1)
        self.just_teleported = max(0, self.just_teleported - 1)

    def _handle_movement(self, movement_action):
        if movement_action == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement_action == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement_action == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement_action == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _handle_paradox_weaving(self, space_held):
        if space_held and self.paradox_cooldown == 0 and self.resources >= self.PARADOX_COST:
            self.resources -= self.PARADOX_COST
            self.paradox_cooldown = self.PARADOX_COOLDOWN_MAX
            new_paradox = {
                "pos": list(self.player_pos),
                "lifetime": self.PARADOX_LIFETIME,
                "creation_step": self.steps
            }
            self.paradoxes.append(new_paradox)
            self._create_particle_burst(self.player_pos, 20, self.COLOR_PARADOX)
            return True
        return False

    def _handle_teleport(self, shift_held):
        if shift_held and self.teleport_cooldown == 0 and self.paradoxes:
            # Teleport to the oldest paradox
            oldest_paradox = min(self.paradoxes, key=lambda p: p["creation_step"])
            
            old_player_pos = list(self.player_pos)
            self.player_pos = list(oldest_paradox["pos"])
            
            self.paradoxes.remove(oldest_paradox)
            
            self.teleport_cooldown = self.TELEPORT_COOLDOWN_MAX
            self.just_teleported = 10 # Visual effect timer
            
            # SFX: Teleport whoosh
            self._create_particle_burst(old_player_pos, 30, self.COLOR_PLAYER, speed_mult=2.0)
            self._create_particle_burst(self.player_pos, 50, self.COLOR_PLAYER, speed_mult=3.0)
            
            return 1.0 # Teleport reward
        return 0.0

    def _update_enemies(self):
        for enemy in self.enemies:
            # Check if slowed by any paradox
            is_slowed = False
            for paradox in self.paradoxes:
                dist_sq = (enemy["pos"][0] - paradox["pos"][0])**2 + (enemy["pos"][1] - paradox["pos"][1])**2
                if dist_sq < self.PARADOX_RADIUS**2:
                    is_slowed = True
                    break
            
            speed = self.ENEMY_BASE_SPEED * self.enemy_speed_multiplier
            if is_slowed:
                speed *= 0.3 # Slowdown factor

            # Move towards player
            direction = [self.player_pos[0] - enemy["pos"][0], self.player_pos[1] - enemy["pos"][1]]
            dist = math.hypot(*direction)
            if dist > 1:
                direction = [d / dist for d in direction]
            
            enemy["pos"][0] += direction[0] * speed
            enemy["pos"][1] += direction[1] * speed

    def _update_paradoxes(self):
        stability_loss = 0
        for p in self.paradoxes:
            p["lifetime"] -= 1
        
        original_count = len(self.paradoxes)
        self.paradoxes = [p for p in self.paradoxes if p["lifetime"] > 0]
        expired_count = original_count - len(self.paradoxes)
        if expired_count > 0:
            # SFX: Paradox fizzle out
            stability_loss += expired_count * 0.5 # Small stability penalty for letting them expire
        return stability_loss

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1
            p["radius"] *= 0.98
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _handle_enemy_paradox_collisions(self):
        stability_loss = 0
        for enemy in self.enemies:
            for paradox in self.paradoxes:
                dist_sq = (enemy["pos"][0] - paradox["pos"][0])**2 + (enemy["pos"][1] - paradox["pos"][1])**2
                if dist_sq < (self.ENEMY_RADIUS + 5)**2: # Use a small buffer
                    paradox["lifetime"] -= 20 # Damage the paradox
                    stability_loss += self.STABILITY_DECAY_ENEMY_CONTACT
                    self._create_particle_burst(enemy["pos"], 5, self.COLOR_ENEMY, speed_mult=0.5)
                    # SFX: Paradox hit feedback
        return stability_loss

    def _handle_player_enemy_collisions(self):
        stability_loss = 0
        for enemy in self.enemies:
            dist_sq = (enemy["pos"][0] - self.player_pos[0])**2 + (enemy["pos"][1] - self.player_pos[1])**2
            if dist_sq < (self.ENEMY_RADIUS + self.PLAYER_RADIUS)**2:
                stability_loss += self.STABILITY_DECAY_PLAYER_CONTACT
                self._create_particle_burst(self.player_pos, 40, self.COLOR_ENEMY, speed_mult=1.5)
                # SFX: Player hit/damage sound
        return stability_loss

    def _handle_player_resource_collisions(self):
        reward = 0
        collected_nodes = []
        for node in self.resource_nodes:
            dist_sq = (node["pos"][0] - self.player_pos[0])**2 + (node["pos"][1] - self.player_pos[1])**2
            if dist_sq < (self.RESOURCE_RADIUS + self.PLAYER_RADIUS)**2:
                collected_nodes.append(node)
                self.resources += 1
                reward += 1.0 # Resource collection reward
                self._create_particle_burst(node["pos"], 15, self.COLOR_RESOURCE)
                # SFX: Resource collected
        
        if collected_nodes:
            self.resource_nodes = [n for n in self.resource_nodes if n not in collected_nodes]
        return reward

    def _update_progression(self):
        # Increase enemy speed
        if self.steps > 0 and self.steps % 50 == 0:
            self.enemy_speed_multiplier += 0.05
        # Increase number of enemies
        if self.steps > 0 and self.steps % 200 == 0:
            self.num_enemies_to_spawn = min(10, self.num_enemies_to_spawn + 1)

    def _spawn_enemies(self):
        while len(self.enemies) < self.num_enemies_to_spawn:
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = [self.np_random.uniform(0, self.WIDTH), -self.ENEMY_RADIUS]
            elif edge == 1: # Bottom
                pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ENEMY_RADIUS]
            elif edge == 2: # Left
                pos = [-self.ENEMY_RADIUS, self.np_random.uniform(0, self.HEIGHT)]
            else: # Right
                pos = [self.WIDTH + self.ENEMY_RADIUS, self.np_random.uniform(0, self.HEIGHT)]
            self.enemies.append({"pos": pos})

    def _spawn_resource(self):
        pos = [
            self.np_random.uniform(self.RESOURCE_RADIUS, self.WIDTH - self.RESOURCE_RADIUS),
            self.np_random.uniform(self.RESOURCE_RADIUS, self.HEIGHT - self.RESOURCE_RADIUS)
        ]
        self.resource_nodes.append({"pos": pos})
        
    def _create_particle_burst(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(1, 4)
            self.particles.append({"pos": list(pos), "vel": vel, "lifetime": lifetime, "color": color, "radius": radius})

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_resources()
        self._render_paradoxes()
        self._render_enemies()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

    def _render_glow(self, pos, radius, color, alpha):
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*color, alpha), (radius, radius), radius)
        self.screen.blit(temp_surf, (int(pos[0] - radius), int(pos[1] - radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_resources(self):
        for node in self.resource_nodes:
            pos_int = (int(node["pos"][0]), int(node["pos"][1]))
            self._render_glow(pos_int, self.RESOURCE_RADIUS * 2.5, self.COLOR_RESOURCE_GLOW, 80)
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, self.RESOURCE_RADIUS, self.COLOR_RESOURCE)
            pygame.gfxdraw.aacircle(self.screen, *pos_int, self.RESOURCE_RADIUS, self.COLOR_RESOURCE)

    def _render_paradoxes(self):
        for paradox in self.paradoxes:
            pos_int = (int(paradox["pos"][0]), int(paradox["pos"][1]))
            # Pulsating Area of Effect
            pulse = (math.sin(self.steps * 0.1 + paradox["creation_step"]) + 1) / 2
            aoe_alpha = int(30 + pulse * 40)
            aoe_color = (*self.COLOR_PARADOX_GLOW, aoe_alpha)
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, self.PARADOX_RADIUS, aoe_color)
            pygame.gfxdraw.aacircle(self.screen, *pos_int, self.PARADOX_RADIUS, aoe_color)
            
            # Core
            core_radius = 5
            self._render_glow(pos_int, core_radius * 3, self.COLOR_PARADOX, 150)
            pygame.gfxdraw.filled_circle(self.screen, *pos_int, core_radius, self.COLOR_PARADOX)
            pygame.gfxdraw.aacircle(self.screen, *pos_int, core_radius, self.COLOR_PARADOX)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            size = self.ENEMY_RADIUS
            rect = pygame.Rect(pos_int[0] - size, pos_int[1] - size, size * 2, size * 2)
            self._render_glow(pos_int, size * 2, self.COLOR_ENEMY_GLOW, 100)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)

    def _render_player(self):
        pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Teleport visual effect
        if self.just_teleported > 0:
            tele_alpha = int(255 * (self.just_teleported / 10))
            tele_radius = int(self.PLAYER_RADIUS * (1 + (10 - self.just_teleported)))
            self._render_glow(pos_int, tele_radius, self.COLOR_PLAYER_GLOW, tele_alpha // 2)

        self._render_glow(pos_int, self.PLAYER_RADIUS * 3, self.COLOR_PLAYER_GLOW, 120)
        pygame.gfxdraw.filled_circle(self.screen, *pos_int, self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, *pos_int, self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 30))
            color = (*p["color"], alpha)
            pos_int = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(p["radius"])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, *pos_int, radius, color)

    def _render_ui(self):
        # Stability Bar
        bar_width = 200
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = 10
        
        stability_ratio = self.paradox_stability / 100.0
        
        if stability_ratio > 0.6:
            bar_color = self.COLOR_STABILITY_BAR_HIGH
        elif stability_ratio > 0.3:
            bar_color = self.COLOR_STABILITY_BAR_MID
        else:
            bar_color = self.COLOR_STABILITY_BAR_LOW
            
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, bar_width * stability_ratio, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)
        
        stability_text = self.font_ui.render("STABILITY", True, self.COLOR_UI_TEXT)
        self.screen.blit(stability_text, (bar_x - stability_text.get_width() - 10, bar_y - 2))

        # Resource Count
        resource_text = self.font_ui.render(f"RESOURCES: {self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(resource_text, (bar_x + bar_width + 10, bar_y - 2))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

    # --- Gymnasium Interface ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "paradox_stability": self.paradox_stability,
            "resources": self.resources,
            "enemies": len(self.enemies),
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It will not run in the headless evaluation environment
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Temporal Paradox")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Arrows: Move")
    print("Space: Weave Paradox")
    print("Shift: Teleport")
    print("--------------------")

    while not terminated:
        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)

    print("\n--- Game Over ---")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Survived for {info['steps']} steps.")
    
    env.close()