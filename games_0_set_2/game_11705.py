import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:26:53.281175
# Source Brief: brief_01705.md
# Brief Index: 1705
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
        "Build and expand your space station by collecting floating resources. "
        "Defend your base from incoming enemy patrols by strategically placing new modules."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to build new modules. Hold Shift while building for "
        "hardened, more durable modules. Press Space to flip the direction of gravity to gather resources."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000
    VICTORY_MODULE_COUNT = 20

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_STAR = (100, 100, 120)
    COLOR_TEXT = (220, 220, 255)
    COLOR_UI_BG = (30, 40, 60, 180)
    COLOR_HEALTH_BAR = (40, 200, 80)
    COLOR_HEALTH_BAR_MID = (250, 220, 50)
    COLOR_HEALTH_BAR_LOW = (220, 50, 50)
    COLOR_HEALTH_BAR_BG = (60, 60, 80)
    
    # Player Base
    BASE_MODULE_SIZE = 20
    BASE_MODULE_HEALTH = 100
    BASE_MODULE_COST = 10
    BASE_MODULE_COLOR = (0, 255, 150)
    BASE_MODULE_GLOW = (0, 255, 150, 50)
    HARDENED_MODULE_HEALTH = 250
    HARDENED_MODULE_COST = 25
    HARDENED_MODULE_COLOR = (0, 180, 255)
    HARDENED_MODULE_GLOW = (0, 180, 255, 60)
    HARDENED_UNLOCK_COUNT = 10

    # Resources
    RESOURCE_COUNT = 50
    RESOURCE_SIZE = 6
    RESOURCE_COLOR = (255, 220, 0)
    RESOURCE_GLOW = (255, 220, 0, 80)
    GRAVITY_STRENGTH = 0.05
    MINING_RADIUS = 30
    
    # Patrols
    PATROL_SIZE = 15
    PATROL_COLOR = (255, 50, 50)
    PATROL_GLOW = (255, 50, 50, 70)
    PATROL_INITIAL_SPEED = 1.0
    PATROL_SPEED_INCREASE_INTERVAL = 200
    PATROL_SPEED_INCREASE_AMOUNT = 0.05
    PATROL_SPAWN_INTERVAL = 500
    PATROL_DANGER_RADIUS = 150

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.render_mode = render_mode
        self._initialize_stars()
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_modules = pygame.sprite.Group()
        self.resources = pygame.sprite.Group()
        self.patrols = pygame.sprite.Group()
        self.vfx_particles = []
        self.last_built_module = None
        self.resource_count = 0
        self.crew_count = 0
        self.gravity_angle = math.pi / 2  # Downwards
        self.last_space_held = False
        self.patrol_speed = self.PATROL_INITIAL_SPEED
        self.next_patrol_spawn_step = self.PATROL_SPAWN_INTERVAL
        
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False
        
        self.base_modules.empty()
        self.resources.empty()
        self.patrols.empty()
        self.vfx_particles.clear()
        
        self.resource_count = 20  # Start with enough to build
        self.crew_count = 1
        
        # Create initial base module (core)
        core_module = Module(self.WIDTH // 2, self.HEIGHT // 2, is_core=True)
        self.base_modules.add(core_module)
        self.last_built_module = core_module
        
        self._spawn_resources(self.RESOURCE_COUNT)
        self._spawn_patrol()
        
        self.gravity_angle = math.pi / 2
        self.last_space_held = False
        self.patrol_speed = self.PATROL_INITIAL_SPEED
        self.next_patrol_spawn_step = self.PATROL_SPAWN_INTERVAL
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0

        # --- Handle Actions ---
        # 1. Build Action
        if movement != 0:
            build_reward = self._handle_build(movement, shift_held)
            reward += build_reward

        # 2. Gravity Flip Action (on press, not hold)
        if space_held and not self.last_space_held:
            self._flip_gravity()
        self.last_space_held = space_held

        # --- Update Game State ---
        update_rewards = self._update_game_state()
        reward += update_rewards
        
        # --- Calculate Step-based Rewards ---
        for patrol in self.patrols:
            if not self.base_modules: break
            dist_to_base = min(math.hypot(patrol.pos.x - m.rect.centerx, patrol.pos.y - m.rect.centery) for m in self.base_modules)
            if dist_to_base < self.PATROL_DANGER_RADIUS:
                reward -= 0.1

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self._get_total_base_health() <= 0:
                reward -= 100 # Defeat
            elif len(self.base_modules) >= self.VICTORY_MODULE_COUNT:
                reward += 100 # Victory
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_build(self, direction, is_hardened_build):
        cost = self.HARDENED_MODULE_COST if is_hardened_build else self.BASE_MODULE_COST
        can_build_hardened = len(self.base_modules) >= self.HARDENED_UNLOCK_COUNT

        if is_hardened_build and not can_build_hardened:
            return 0 # Cannot build hardened yet

        if self.resource_count >= cost and self.last_built_module:
            offset = self.BASE_MODULE_SIZE
            target_pos = self.last_built_module.rect.center
            if direction == 1: target_pos = (target_pos[0], target_pos[1] - offset)
            elif direction == 2: target_pos = (target_pos[0], target_pos[1] + offset)
            elif direction == 3: target_pos = (target_pos[0] - offset, target_pos[1])
            elif direction == 4: target_pos = (target_pos[0] + offset, target_pos[1])

            # Check for validity (in bounds and not overlapping)
            new_rect = pygame.Rect(0, 0, self.BASE_MODULE_SIZE, self.BASE_MODULE_SIZE)
            new_rect.center = target_pos
            if self.screen.get_rect().contains(new_rect) and not any(m.rect.colliderect(new_rect) for m in self.base_modules):
                self.resource_count -= cost
                new_module = Module(target_pos[0], target_pos[1], is_hardened=is_hardened_build, parent=self.last_built_module)
                self.base_modules.add(new_module)
                self.last_built_module = new_module
                self.crew_count += 1
                # Visual effect for building
                self._create_vfx('build', new_module.rect.center, 10, new_module.color)
                return 5 # Build reward
        return 0

    def _flip_gravity(self):
        self.gravity_angle = (self.gravity_angle + math.pi) % (2 * math.pi)
        # sfx: "gravity_shift.wav"
        self._create_vfx('shockwave', (self.WIDTH/2, self.HEIGHT/2), 15)

    def _update_game_state(self):
        reward = 0
        
        # Update entities
        gravity_vec = pygame.Vector2(math.cos(self.gravity_angle), math.sin(self.gravity_angle)) * self.GRAVITY_STRENGTH
        self.resources.update(gravity_vec)
        self.patrols.update(self.patrol_speed, self.base_modules)
        
        # Update VFX
        for particle in self.vfx_particles[:]:
            particle.update()
            if particle.is_dead():
                self.vfx_particles.remove(particle)
        
        # Collisions: Resources vs Base
        for module in self.base_modules:
            if pygame.sprite.spritecollide(module, self.resources, True, pygame.sprite.collide_circle_ratio(self.MINING_RADIUS / self.BASE_MODULE_SIZE)):
                self.resource_count += 1
                reward += 1
                # sfx: "resource_collect.wav"
                self._create_vfx('collect', module.rect.center, 8, self.RESOURCE_COLOR)

        # Collisions: Patrols vs Base
        for patrol in self.patrols:
            collided_modules = pygame.sprite.spritecollide(patrol, self.base_modules, False)
            for module in collided_modules:
                is_invulnerable = module.is_core and len(self.base_modules) == 1
                if not is_invulnerable:
                    module.take_damage(25)
                    # sfx: "impact.wav"
                    self._create_vfx('explosion', module.rect.center, 12, self.PATROL_COLOR)
                    if module.health <= 0:
                        reward -= 2 # Module destroyed reward
                        self.crew_count -= 1
                        # sfx: "explosion_large.wav"
                        self._create_vfx('explosion', module.rect.center, 20, module.color)
                        module.kill()
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % self.PATROL_SPEED_INCREASE_INTERVAL == 0:
            self.patrol_speed += self.PATROL_SPEED_INCREASE_AMOUNT
        if self.steps >= self.next_patrol_spawn_step:
            self._spawn_patrol()
            self.next_patrol_spawn_step += self.PATROL_SPAWN_INTERVAL
            
        return reward

    def _check_termination(self):
        if self._get_total_base_health() <= 0: return True
        if len(self.base_modules) >= self.VICTORY_MODULE_COUNT: return True
        return False
        
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resource_count": self.resource_count,
            "crew_count": self.crew_count,
            "base_size": len(self.base_modules),
            "base_health": self._get_total_base_health()
        }

    def _get_total_base_health(self):
        return sum(m.health for m in self.base_modules)

    def _get_max_base_health(self):
        return sum(m.max_health for m in self.base_modules)

    def _initialize_stars(self):
        self.stars = []
        for i in range(200):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.uniform(0.5, 1.5)
            self.stars.append({'pos': pygame.Vector2(x, y), 'size': size})

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Parallax stars
        for star in self.stars:
            star['pos'].y = (star['pos'].y + 0.1 * star['size']) % self.HEIGHT
            pygame.draw.circle(self.screen, self.COLOR_STAR, (int(star['pos'].x), int(star['pos'].y)), star['size'])

    def _render_game(self):
        # Draw connections between modules
        for module in self.base_modules:
            if module.parent:
                pygame.draw.line(self.screen, (80, 100, 120), module.rect.center, module.parent.rect.center, 2)
        
        # Use custom draw methods
        for r in self.resources: r.draw(self.screen)
        for m in self.base_modules: m.draw(self.screen)
        for p in self.patrols: p.draw(self.screen)
        
        for particle in self.vfx_particles:
            particle.draw(self.screen)
            
        # Draw gravity indicator
        center = (self.WIDTH // 2, self.HEIGHT // 2)
        arrow_len = 30
        end_pos = (center[0] + arrow_len * math.cos(self.gravity_angle),
                   center[1] + arrow_len * math.sin(self.gravity_angle))
        
        pygame.draw.line(self.screen, (255, 255, 255, 50), center, end_pos, 3)
        # Arrowhead
        p1 = (end_pos[0] - 8 * math.cos(self.gravity_angle - 0.5), end_pos[1] - 8 * math.sin(self.gravity_angle - 0.5))
        p2 = (end_pos[0] - 8 * math.cos(self.gravity_angle + 0.5), end_pos[1] - 8 * math.sin(self.gravity_angle + 0.5))
        pygame.draw.aaline(self.screen, (255, 255, 255, 50), end_pos, p1)
        pygame.draw.aaline(self.screen, (255, 255, 255, 50), end_pos, p2)


    def _render_ui(self):
        # UI Panel Top Left
        res_text = self.font_main.render(f"RES: {self.resource_count}", True, self.COLOR_TEXT)
        crew_text = self.font_main.render(f"CREW: {self.crew_count}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (10, 10))
        self.screen.blit(crew_text, (10, 30))

        # Health Bar Top Right
        max_health = self._get_max_base_health()
        current_health = self._get_total_base_health()
        health_ratio = current_health / max_health if max_health > 0 else 0
        
        bar_width = 150
        bar_height = 20
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        
        health_color = self.COLOR_HEALTH_BAR
        if health_ratio < 0.66: health_color = self.COLOR_HEALTH_BAR_MID
        if health_ratio < 0.33: health_color = self.COLOR_HEALTH_BAR_LOW
        
        current_bar_width = max(0, int(bar_width * health_ratio))
        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, current_bar_width, bar_height))
        
        health_text = self.font_main.render("BASE HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (bar_x, bar_y + bar_height + 5))

        # Game Over / Victory Text
        if self.game_over:
            if self._get_total_base_health() <= 0:
                end_text = self.font_large.render("BASE DESTROYED", True, self.COLOR_HEALTH_BAR_LOW)
            elif len(self.base_modules) >= self.VICTORY_MODULE_COUNT:
                end_text = self.font_large.render("VICTORY", True, self.COLOR_HEALTH_BAR)
            else: # Truncated
                end_text = self.font_large.render("TIME UP", True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _spawn_resources(self, count):
        for _ in range(count):
            pos = (random.randint(20, self.WIDTH - 20), random.randint(20, self.HEIGHT - 20))
            # Ensure not spawning on base
            if not any(m.rect.collidepoint(pos) for m in self.base_modules):
                self.resources.add(ResourceParticle(pos[0], pos[1]))

    def _spawn_patrol(self):
        side = random.choice(['left', 'right', 'top', 'bottom'])
        if side == 'left':
            pos = (-self.PATROL_SIZE, random.randint(0, self.HEIGHT))
            vel = (1, 0)
        elif side == 'right':
            pos = (self.WIDTH + self.PATROL_SIZE, random.randint(0, self.HEIGHT))
            vel = (-1, 0)
        elif side == 'top':
            pos = (random.randint(0, self.WIDTH), -self.PATROL_SIZE)
            vel = (0, 1)
        else: # bottom
            pos = (random.randint(0, self.WIDTH), self.HEIGHT + self.PATROL_SIZE)
            vel = (0, -1)
        self.patrols.add(PatrolShip(pos[0], pos[1], vel[0], vel[1]))

    def _create_vfx(self, vfx_type, pos, count, color=(255,255,255)):
        if vfx_type == 'explosion':
            for _ in range(count):
                self.vfx_particles.append(VFXParticle(pos, lifespan=20, size=random.uniform(1, 3), color=color))
        elif vfx_type == 'collect':
            for _ in range(count):
                self.vfx_particles.append(VFXParticle(pos, lifespan=10, size=random.uniform(1, 2), speed_mult=0.5, color=color))
        elif vfx_type == 'build':
             for _ in range(count):
                self.vfx_particles.append(VFXParticle(pos, lifespan=15, size=random.uniform(1, 2.5), speed_mult=1.5, color=color))
        elif vfx_type == 'shockwave':
            self.vfx_particles.append(VFXParticle(pos, vfx_type='shockwave', lifespan=15))

    def close(self):
        pygame.quit()


# --- Helper Classes ---

class Module(pygame.sprite.Sprite):
    def __init__(self, x, y, is_hardened=False, is_core=False, parent=None):
        super().__init__()
        self.size = GameEnv.BASE_MODULE_SIZE
        self.is_hardened = is_hardened
        self.is_core = is_core
        self.parent = parent
        
        if is_hardened:
            self.max_health = GameEnv.HARDENED_MODULE_HEALTH
            self.color = GameEnv.HARDENED_MODULE_COLOR
            self.glow_color = GameEnv.HARDENED_MODULE_GLOW
        else:
            self.max_health = GameEnv.BASE_MODULE_HEALTH
            self.color = GameEnv.BASE_MODULE_COLOR
            self.glow_color = GameEnv.BASE_MODULE_GLOW
        
        self.health = self.max_health
        self.image = pygame.Surface((self.size, self.size)) # Not used for drawing
        self.rect = self.image.get_rect(center=(x, y))
        self.last_damage_time = -100

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)
        self.last_damage_time = pygame.time.get_ticks()

    def draw(self, surface):
        # Flashing effect when damaged
        if pygame.time.get_ticks() - self.last_damage_time < 150:
            if (pygame.time.get_ticks() // 50) % 2 == 0:
                return # Skip drawing to flash
        
        # Draw glow
        glow_rect = self.rect.inflate(self.size * 0.8, self.size * 0.8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.glow_color, glow_surf.get_rect(), border_radius=5)
        surface.blit(glow_surf, glow_rect.topleft)

        # Draw main rect
        pygame.draw.rect(surface, self.color, self.rect, border_radius=3)
        pygame.draw.rect(surface, tuple(min(255, c + 50) for c in self.color), self.rect, 2, border_radius=3)

        # Draw core indicator
        if self.is_core:
            pygame.draw.circle(surface, (255, 255, 255), self.rect.center, 3)

    def update(self):
        pass # Modules are static

class ResourceParticle(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.size = GameEnv.RESOURCE_SIZE
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.image = pygame.Surface((self.size * 2, self.size * 2)) # Not used
        self.rect = self.image.get_rect(center=(x, y))
        self.color = GameEnv.RESOURCE_COLOR
        self.glow_color = GameEnv.RESOURCE_GLOW

    def update(self, gravity):
        self.vel += gravity
        self.pos += self.vel
        
        # Dampening
        self.vel *= 0.98

        # Screen boundary bounce
        if self.pos.x < 0 or self.pos.x > GameEnv.WIDTH: self.vel.x *= -0.8
        if self.pos.y < 0 or self.pos.y > GameEnv.HEIGHT: self.vel.y *= -0.8
        self.pos.x = np.clip(self.pos.x, 0, GameEnv.WIDTH)
        self.pos.y = np.clip(self.pos.y, 0, GameEnv.HEIGHT)
        
        self.rect.center = self.pos

    def draw(self, surface):
        # Glow
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.size, self.glow_color)
        # Core
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.size - 2, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.size - 2, self.color)

class PatrolShip(pygame.sprite.Sprite):
    def __init__(self, x, y, vx, vy):
        super().__init__()
        self.size = GameEnv.PATROL_SIZE
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(vx, vy).normalize()
        self.image = pygame.Surface((self.size * 2, self.size * 2)) # Not used
        self.rect = self.image.get_rect(center=(x, y))
        self.color = GameEnv.PATROL_COLOR
        self.glow_color = GameEnv.PATROL_GLOW
        self.thruster_particles = deque(maxlen=20)

    def update(self, speed, base_modules):
        self.pos += self.vel * speed
        self.rect.center = self.pos

        # Thruster VFX
        self.thruster_particles.append(self.pos - self.vel * self.size * 0.7)

        # Wrap around screen
        margin = self.size * 2
        if self.pos.x < -margin: self.pos.x = GameEnv.WIDTH + margin
        if self.pos.x > GameEnv.WIDTH + margin: self.pos.x = -margin
        if self.pos.y < -margin: self.pos.y = GameEnv.HEIGHT + margin
        if self.pos.y > GameEnv.HEIGHT + margin: self.pos.y = -margin

    def draw(self, surface):
        # Thruster trail
        for i, p in enumerate(self.thruster_particles):
            alpha = int(255 * (i / len(self.thruster_particles)))
            size = 1 + (i / len(self.thruster_particles)) * 2
            pygame.draw.circle(surface, (*self.color, alpha), p, size)

        # Body
        angle = self.vel.angle_to(pygame.Vector2(1, 0))
        points = []
        for i in range(3):
            rad = math.radians(angle + i * 120)
            p = (self.pos.x + self.size * math.cos(rad),
                 self.pos.y - self.size * math.sin(rad))
            points.append(p)
        
        pygame.gfxdraw.aapolygon(surface, points, self.glow_color)
        pygame.gfxdraw.filled_polygon(surface, points, self.color)

class VFXParticle:
    def __init__(self, pos, vfx_type='spark', lifespan=20, size=2, speed_mult=1.0, color=(255, 255, 255)):
        self.pos = pygame.Vector2(pos)
        self.vfx_type = vfx_type
        self.lifespan = lifespan
        self.initial_lifespan = lifespan
        self.size = size
        self.color = color
        
        if self.vfx_type == 'spark':
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        elif self.vfx_type == 'shockwave':
            self.radius = 0
            self.max_radius = 100
            self.vel = pygame.Vector2(0,0) # Not used for movement

    def update(self):
        if self.vfx_type == 'spark':
            self.pos += self.vel
            self.vel *= 0.9 # friction
        elif self.vfx_type == 'shockwave':
            self.radius += self.max_radius / self.initial_lifespan

        self.lifespan -= 1
        
    def is_dead(self):
        return self.lifespan <= 0

    def draw(self, surface):
        life_ratio = self.lifespan / self.initial_lifespan
        current_color = (*self.color, int(255 * life_ratio))
        
        if self.vfx_type == 'spark':
            current_size = self.size * life_ratio
            pygame.draw.circle(surface, current_color, self.pos, current_size)
        elif self.vfx_type == 'shockwave':
            if self.radius > 0:
                pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), current_color)


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Space Base Sim")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # --- Action Mapping for Human Play ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        # Poll for events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Get key states for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        if not done:
            # For human play, we might want to step on every frame regardless of action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to get it into a format Pygame can display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        # Display score for human player
        score_text = env.font_main.render(f"Total Reward: {total_reward:.1f}", True, (255, 255, 255))
        screen.blit(score_text, (10, GameEnv.HEIGHT - 30))
        
        pygame.display.flip()
        env.clock.tick(GameEnv.FPS)

    env.close()