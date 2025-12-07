import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:04:17.556489
# Source Brief: brief_03247.md
# Brief Index: 3247
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes ---

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, vel, color, lifetime, size, gravity=0.1):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.size = size
        self.gravity = gravity

    def update(self):
        self.pos += self.vel
        self.vel.y += self.gravity
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            color = (*self.color, alpha)
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (self.size, self.size), self.size)
            surface.blit(temp_surf, self.pos - pygame.Vector2(self.size, self.size), special_flags=pygame.BLEND_RGBA_ADD)

class Projectile:
    """A spore projectile fired by a colony."""
    def __init__(self, pos, target_pos, p_type, speed, damage):
        self.pos = pygame.Vector2(pos)
        self.type = p_type
        self.speed = speed
        self.damage = damage
        self.color = GameEnv.COLOR_PLAYER_SPORE if p_type == 'player' else GameEnv.COLOR_ENEMY_SPORE
        
        direction = (target_pos - pos)
        if direction.length() > 0:
            self.vel = direction.normalize() * self.speed
        else:
            self.vel = pygame.Vector2(0, -self.speed)
        self.lifetime = 200 # steps to live

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1

    def draw(self, surface):
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), 3, self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), 3, self.color)

class Colony:
    """Represents a fungal colony."""
    def __init__(self, pos, c_type, initial_radius=15):
        self.pos = pygame.Vector2(pos)
        self.type = c_type
        self.radius = float(initial_radius)
        self.target_radius = float(initial_radius)
        self.max_radius = 60.0
        
        self.health = 100.0
        self.max_health = 100.0
        
        self.color = GameEnv.COLOR_PLAYER if c_type == 'player' else GameEnv.COLOR_ENEMY
        self.glow_color = GameEnv.COLOR_PLAYER_GLOW if c_type == 'player' else GameEnv.COLOR_ENEMY_GLOW
        
        self.shoot_cooldown = random.randint(30, 60)
        self.target_colony = None
        self.pulse_timer = random.uniform(0, 2 * math.pi)

    def update(self, all_colonies, projectiles, p_speed, p_damage, growth_boost):
        # Growth
        growth_rate = 0.05 + (0.15 if growth_boost and self.type == 'player' else 0)
        if self.target_radius < self.max_radius:
            self.target_radius += growth_rate
        self.radius += (self.target_radius - self.radius) * 0.1 # Smooth growth

        # Health Regen
        self.health = min(self.max_health, self.health + 0.05)

        # Combat Logic
        self.shoot_cooldown -= 1
        if self.shoot_cooldown <= 0:
            self.find_target(all_colonies)
            if self.target_colony:
                # sfx: spore_shoot
                projectiles.append(Projectile(self.pos, self.target_colony.pos, self.type, p_speed, p_damage))
                self.shoot_cooldown = 100 - int(self.radius / 2) # Faster shoot rate for bigger colonies
        
        self.pulse_timer += 0.05

    def find_target(self, all_colonies):
        closest_enemy = None
        min_dist = float('inf')
        for colony in all_colonies:
            if colony.type != self.type:
                dist = self.pos.distance_to(colony.pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy = colony
        self.target_colony = closest_enemy

    def take_damage(self, amount):
        self.health -= amount
        return self.health <= 0

    def draw(self, surface):
        # Glow effect
        glow_radius = int(self.radius * 1.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pulse = (math.sin(self.pulse_timer) + 1) / 2 # 0 to 1
        alpha = 50 + int(pulse * 50)
        pygame.draw.circle(glow_surf, (*self.glow_color, alpha), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Main body
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color)
        
        # Health bar
        if self.health < self.max_health:
            bar_width = 40
            bar_height = 5
            bar_x = self.pos.x - bar_width / 2
            bar_y = self.pos.y - self.radius - 15
            health_pct = max(0, self.health / self.max_health)
            
            pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(surface, self.color, (bar_x, bar_y, bar_width * health_pct, bar_height))


# --- Gymnasium Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a fungal colony, spreading across the terrain to achieve dominance. "
        "Deploy new colonies and fight off rivals with spores."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the deployment target. "
        "Press space to create a new colony and shift to activate a growth boost."
    )
    auto_advance = True

    # --- Colors and Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYABLE_WIDTH, PLAYABLE_HEIGHT = 600, 320
    X_OFFSET = (SCREEN_WIDTH - PLAYABLE_WIDTH) // 2
    Y_OFFSET = (SCREEN_HEIGHT - PLAYABLE_HEIGHT) // 2
    
    COLOR_BG = (25, 20, 20)
    COLOR_WATER = (50, 80, 150)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 150, 70)
    COLOR_PLAYER_SPORE = (180, 255, 200)
    COLOR_ENEMY = (255, 60, 60)
    COLOR_ENEMY_GLOW = (150, 40, 40)
    COLOR_ENEMY_SPORE = (255, 150, 150)
    COLOR_TARGET = (0, 220, 220)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BAR = (0, 180, 100)
    COLOR_UI_BAR_BG = (60, 60, 60)

    MAX_STEPS = 2000
    WIN_BIOMASS_PCT = 0.90
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 24, bold=True)
        
        self.render_mode = render_mode

        # State variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_colonies = []
        self.enemy_colonies = []
        self.projectiles = []
        self.particles = []
        self.water_bodies = []
        self.deployment_target = pygame.Vector2(0, 0)
        self.last_space_held = False
        self.last_shift_held = False
        self.ability_cooldown = 0
        self.ability_active_timer = 0
        self.next_enemy_spawn_step = 0
        self.next_enemy_buff_step = 0
        self.enemy_spore_speed = 0
        self.player_spore_speed = 0
        self.player_spore_damage = 0
        self.enemy_spore_damage = 0
        self.last_biomass_pct = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.deployment_target = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        self._generate_terrain()

        self.player_colonies = [Colony(pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2), 'player')]
        self.enemy_colonies = []
        self._spawn_enemy()
        self._spawn_enemy()
        
        self.projectiles = []
        self.particles = []
        
        self.last_space_held = False
        self.last_shift_held = False
        self.ability_cooldown = 0
        self.ability_active_timer = 0
        
        self.next_enemy_spawn_step = 200
        self.next_enemy_buff_step = 300
        self.enemy_spore_speed = 1.5
        self.player_spore_speed = 2.0
        self.player_spore_damage = 2.0
        self.enemy_spore_damage = 1.8

        self.last_biomass_pct = self._calculate_biomass_pct()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        self._handle_input(action)
        reward += self._update_game_state()
        
        self.steps += 1
        
        current_biomass_pct = self._calculate_biomass_pct()
        biomass_delta = current_biomass_pct - self.last_biomass_pct
        if biomass_delta > 0:
            reward += biomass_delta * 100 * 0.1 # +0.1 per 1% increase
        elif biomass_delta < 0:
            reward += biomass_delta * 100 * 0.2 # -0.2 per 1% decrease
        self.last_biomass_pct = current_biomass_pct
        self.score += reward

        terminated = self._check_termination()
        if terminated:
            if not self.player_colonies:
                reward -= 100 # Loss
            elif current_biomass_pct >= self.WIN_BIOMASS_PCT:
                reward += 100 # Victory
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Move deployment target
        target_speed = 8
        if movement == 1: self.deployment_target.y -= target_speed
        elif movement == 2: self.deployment_target.y += target_speed
        elif movement == 3: self.deployment_target.x -= target_speed
        elif movement == 4: self.deployment_target.x += target_speed
        
        self.deployment_target.x = np.clip(self.deployment_target.x, self.X_OFFSET, self.SCREEN_WIDTH - self.X_OFFSET)
        self.deployment_target.y = np.clip(self.deployment_target.y, self.Y_OFFSET, self.SCREEN_HEIGHT - self.Y_OFFSET)

        # Deploy spore on space press
        if space_held and not self.last_space_held:
            self._deploy_spore()
        
        # Activate ability on shift press
        if shift_held and not self.last_shift_held and self.ability_cooldown <= 0:
            # sfx: ability_activate
            self.ability_active_timer = 150 # 5 seconds at 30fps
            self.ability_cooldown = 600 # 20 seconds cooldown
            self._create_particles(self.deployment_target, 50, self.COLOR_PLAYER, 2.0, 50)

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_game_state(self):
        event_reward = 0

        # Update timers
        if self.ability_cooldown > 0: self.ability_cooldown -= 1
        if self.ability_active_timer > 0: self.ability_active_timer -= 1
        
        # Update colonies
        all_colonies = self.player_colonies + self.enemy_colonies
        for colony in all_colonies:
            is_boosted = self.ability_active_timer > 0 and self.deployment_target.distance_to(colony.pos) < 100
            p_speed = self.player_spore_speed if colony.type == 'player' else self.enemy_spore_speed
            p_damage = self.player_spore_damage if colony.type == 'player' else self.enemy_spore_damage
            colony.update(all_colonies, self.projectiles, p_speed, p_damage, is_boosted)

        # Update and collide projectiles
        dead_projectiles = []
        for p in self.projectiles:
            p.update()
            if not (0 <= p.pos.x < self.SCREEN_WIDTH and 0 <= p.pos.y < self.SCREEN_HEIGHT) or p.lifetime <= 0:
                dead_projectiles.append(p)
                continue
            
            target_list = self.enemy_colonies if p.type == 'player' else self.player_colonies
            for colony in target_list:
                if colony.pos.distance_to(p.pos) < colony.radius:
                    # sfx: impact
                    if colony.take_damage(p.damage):
                        if colony.type == 'player':
                            event_reward -= 1.0 # Lost a colony
                        else:
                            event_reward += 5.0 # Destroyed enemy
                    self._create_particles(p.pos, 15, colony.color, 0.5, 30)
                    dead_projectiles.append(p)
                    break
        
        # Clean up dead things
        self.projectiles = [p for p in self.projectiles if p not in dead_projectiles]
        
        destroyed_player_colonies = [c for c in self.player_colonies if c.health <= 0]
        destroyed_enemy_colonies = [c for c in self.enemy_colonies if c.health <= 0]
        
        for c in destroyed_player_colonies + destroyed_enemy_colonies:
            # sfx: colony_destroyed
            self._create_particles(c.pos, 100, c.color, 1.5, 60)
        
        self.player_colonies = [c for c in self.player_colonies if c.health > 0]
        self.enemy_colonies = [c for c in self.enemy_colonies if c.health > 0]

        # Update particles
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

        # Dynamic difficulty
        if self.steps >= self.next_enemy_spawn_step:
            self._spawn_enemy()
            self.next_enemy_spawn_step += 200
        
        if self.steps >= self.next_enemy_buff_step:
            self.enemy_spore_speed += 0.05
            self.next_enemy_buff_step += 300

        return event_reward

    def _deploy_spore(self):
        # Check if deployment is valid
        if any(b.collidepoint(self.deployment_target) for b in self.water_bodies):
            return # Cannot deploy in water
        
        all_colonies = self.player_colonies + self.enemy_colonies
        for colony in all_colonies:
            if colony.pos.distance_to(self.deployment_target) < colony.radius + 10:
                return # Too close to another colony

        # Find nearest player colony to 'fund' the new one
        if not self.player_colonies: return
        
        parent = min(self.player_colonies, key=lambda c: c.pos.distance_to(self.deployment_target))
        
        cost = 30
        if parent.health > cost + 10:
            # sfx: deploy_spore
            parent.health -= cost
            self.player_colonies.append(Colony(pygame.Vector2(self.deployment_target), 'player'))
            self._create_particles(self.deployment_target, 30, self.COLOR_PLAYER, 1.0, 40)

    def _spawn_enemy(self):
        if len(self.enemy_colonies) > 20: return
        
        for _ in range(100): # Try 100 times to find a valid spot
            pos = pygame.Vector2(
                random.uniform(self.X_OFFSET, self.SCREEN_WIDTH - self.X_OFFSET),
                random.uniform(self.Y_OFFSET, self.SCREEN_HEIGHT - self.Y_OFFSET)
            )
            
            if any(b.collidepoint(pos) for b in self.water_bodies): continue
            
            all_colonies = self.player_colonies + self.enemy_colonies
            if not any(c.pos.distance_to(pos) < c.radius + 20 for c in all_colonies):
                self.enemy_colonies.append(Colony(pos, 'enemy'))
                # sfx: enemy_spawn
                self._create_particles(pos, 30, self.COLOR_ENEMY, 1.0, 40)
                return

    def _calculate_biomass_pct(self):
        player_area = sum(math.pi * c.radius**2 for c in self.player_colonies)
        water_area = sum(b.width * b.height for b in self.water_bodies)
        total_playable_area = (self.PLAYABLE_WIDTH * self.PLAYABLE_HEIGHT) - water_area
        if total_playable_area <= 0: return 0.0
        return min(1.0, player_area / total_playable_area)

    def _check_termination(self):
        return (
            self.steps >= self.MAX_STEPS or
            not self.player_colonies or
            self._calculate_biomass_pct() >= self.WIN_BIOMASS_PCT
        )

    def _generate_terrain(self):
        self.water_bodies = []
        num_lakes = self.np_random.integers(3, 6)
        for _ in range(num_lakes):
            w = self.np_random.integers(50, 150)
            h = self.np_random.integers(50, 150)
            x = self.np_random.integers(self.X_OFFSET, self.SCREEN_WIDTH - self.X_OFFSET - w)
            y = self.np_random.integers(self.Y_OFFSET, self.SCREEN_HEIGHT - self.Y_OFFSET - h)
            self.water_bodies.append(pygame.Rect(x, y, w, h))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw water
        for lake in self.water_bodies:
            pygame.draw.rect(self.screen, self.COLOR_WATER, lake, border_radius=15)

        # Draw particles (underneath colonies)
        for p in self.particles:
            p.draw(self.screen)

        # Draw colonies
        all_colonies = self.player_colonies + self.enemy_colonies
        for colony in sorted(all_colonies, key=lambda c: c.radius):
            colony.draw(self.screen)

        # Draw projectiles
        for p in self.projectiles:
            p.draw(self.screen)

        # Draw deployment target
        if not self.game_over:
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            radius = int(8 + pulse * 4)
            alpha = 150 + int(pulse * 105)
            pygame.gfxdraw.aacircle(self.screen, int(self.deployment_target.x), int(self.deployment_target.y), radius, (*self.COLOR_TARGET, alpha))
            pygame.gfxdraw.aacircle(self.screen, int(self.deployment_target.x), int(self.deployment_target.y), radius - 1, (*self.COLOR_TARGET, alpha))

    def _render_ui(self):
        # Biomass Bar
        biomass_pct = self._calculate_biomass_pct()
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, 10, bar_width * biomass_pct, bar_height), border_radius=4)
        biomass_text = self.font_large.render(f"Biomass: {int(biomass_pct * 100)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(biomass_text, (15, 9))

        # Ability Cooldown
        ability_y = self.SCREEN_HEIGHT - 35
        ability_text = self.font_large.render("BOOST", True, self.COLOR_UI_TEXT)
        self.screen.blit(ability_text, (self.SCREEN_WIDTH/2 - ability_text.get_width()/2, ability_y))

        if self.ability_active_timer > 0:
            status_text = "ACTIVE"
            color = self.COLOR_PLAYER
        elif self.ability_cooldown > 0:
            status_text = f"CD: {self.ability_cooldown / 30:.1f}s"
            color = (150, 150, 150)
        else:
            status_text = "READY"
            color = self.COLOR_TARGET
        
        status_surf = self.font_small.render(status_text, True, color)
        self.screen.blit(status_surf, (self.SCREEN_WIDTH/2 - status_surf.get_width()/2, ability_y + 20))

        # Game Over Text
        if self.game_over:
            if not self.player_colonies:
                msg = "EXTINCTION"
                color = self.COLOR_ENEMY
            elif biomass_pct >= self.WIN_BIOMASS_PCT:
                msg = "DOMINANCE ACHIEVED"
                color = self.COLOR_PLAYER
            else:
                msg = "SIMULATION ENDED"
                color = self.COLOR_UI_TEXT
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,180), text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "biomass_pct": self._calculate_biomass_pct(),
            "player_colonies": len(self.player_colonies),
            "enemy_colonies": len(self.enemy_colonies),
        }
        
    def _create_particles(self, pos, count, color, speed_mult, lifetime):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1.0, 3.0) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            size = random.randint(2, 5)
            p_lifetime = random.randint(int(lifetime/2), lifetime)
            self.particles.append(Particle(pos, vel, color, p_lifetime, size))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # The 'dummy' video driver must be unset to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fungal Domination")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Manual Control Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds before reset
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()