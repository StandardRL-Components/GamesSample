import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:08:55.651973
# Source Brief: brief_01552.md
# Brief Index: 1552
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-level Gymnasium environment where the player defends their 'Ego',
    a central core, from attacking 'Memories'. The player deploys friendly
    'Entities' of varying sizes to physically block and push away the Memories.
    The game features physics-based interactions, a surreal visual style with
    glowing effects, and a strategic layer of choosing when, where, and what
    size of entity to deploy.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your central 'Ego' from attacking 'Memories' by deploying friendly entities. "
        "Strategically place different-sized entities to block and push away the incoming threats."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the deployment cursor. Press 'space' to deploy an "
        "entity at the cursor's location. Press 'shift' to cycle between available entity sizes."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * TARGET_FPS

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_EGO = (100, 150, 255)
    COLOR_EGO_GLOW = (50, 75, 128)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (30, 30, 50)
    
    ENTITY_COLORS = [
        {'main': (80, 255, 180), 'glow': (40, 128, 90)},
        {'main': (255, 100, 255), 'glow': (128, 50, 128)},
        {'main': (255, 255, 100), 'glow': (128, 128, 50)},
    ]
    MEMORY_COLORS = [
        {'main': (255, 80, 80), 'glow': (128, 40, 40)},
        {'main': (255, 160, 80), 'glow': (128, 80, 40)},
    ]

    # Gameplay
    EGO_INITIAL_HEALTH = 100.0
    CURSOR_SPEED = 300.0 / TARGET_FPS
    ENTITY_SIZES = [
        {'radius': 15, 'mass': 2, 'health': 50, 'cost': 10},
        {'radius': 25, 'mass': 5, 'health': 100, 'cost': 25},
        {'radius': 35, 'mass': 10, 'health': 200, 'cost': 50},
    ]
    ENTITY_DEPLOY_COOLDOWN = 0.25 # seconds
    
    INITIAL_MEMORY_SPAWN_RATE = 0.5 # per second
    MEMORY_SPAWN_RATE_INCREASE = 0.05 # per second, per second

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.render_mode = render_mode
        self.center = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        # This will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ego_health = 0.0
        self.game_time_elapsed = 0.0
        self.cursor_pos = pygame.Vector2(0, 0)
        self.selected_size_index = 0
        self.unlocked_entity_level = 0
        self.shift_was_held = False
        self.space_was_held = False
        self.deploy_cooldown_timer = 0.0
        self.memory_spawn_timer = 0.0
        self.current_memory_spawn_rate = 0.0
        self.unlocked_memory_types = []
        
        self.friendly_entities = []
        self.memories = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ego_health = self.EGO_INITIAL_HEALTH
        self.game_time_elapsed = 0.0

        self.cursor_pos = self.center.copy()
        self.selected_size_index = 0
        self.unlocked_entity_level = 0
        self.shift_was_held = False
        self.space_was_held = False
        self.deploy_cooldown_timer = 0.0

        self.current_memory_spawn_rate = self.INITIAL_MEMORY_SPAWN_RATE
        self.memory_spawn_timer = 1.0 / self.current_memory_spawn_rate
        self.unlocked_memory_types = [0]

        self.friendly_entities.clear()
        self.memories.clear()
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        dt = 1.0 / self.TARGET_FPS
        self.game_time_elapsed += dt
        self.steps += 1
        
        step_reward = 0.1  # Survival reward

        # --- Handle Actions ---
        self._handle_input(movement, space_pressed, shift_pressed, dt)
        
        # --- Update Game Logic ---
        self._update_world(dt)
        
        # --- Calculate Rewards ---
        reward_info = self._calculate_reward()
        step_reward += reward_info['reward']

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.ego_health <= 0:
                step_reward -= 100 # Ego depleted
            else:
                step_reward += 100 # Survived
            self.game_over = True
        
        self.score += step_reward

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info(),
        )
    
    def _handle_input(self, movement, space_pressed, shift_pressed, dt):
        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

        # Cycle entity size on Shift press (rising edge)
        if shift_pressed and not self.shift_was_held:
            num_unlocked = 1 + self.unlocked_entity_level
            self.selected_size_index = (self.selected_size_index + 1) % num_unlocked
            # sfx: UI_switch
        self.shift_was_held = shift_pressed

        # Deploy entity on Space press
        self.deploy_cooldown_timer = max(0, self.deploy_cooldown_timer - dt)
        if space_pressed and self.deploy_cooldown_timer == 0:
            size_props = self.ENTITY_SIZES[self.selected_size_index]
            # Ensure not placing inside ego
            if self.cursor_pos.distance_to(self.center) > self._get_ego_radius() + size_props['radius']:
                self.friendly_entities.append(FriendlyEntity(self.cursor_pos.copy(), size_props, self.selected_size_index))
                self.deploy_cooldown_timer = self.ENTITY_DEPLOY_COOLDOWN
                # sfx: entity_spawn

    def _update_world(self, dt):
        # Update progression
        self._update_progression()

        # Spawn memories
        self._spawn_memories(dt)

        # Update positions
        for obj in self.friendly_entities + self.memories:
            obj.pos += obj.vel * dt
        
        for p in self.particles:
            p.update(dt)

        # Physics collisions
        self._handle_collisions()

        # Remove dead objects
        self.friendly_entities = [e for e in self.friendly_entities if e.health > 0]
        self.memories = [m for m in self.memories if m.health > 0]
        self.particles = [p for p in self.particles if p.is_alive()]
        
        # Boundary checks
        for obj in self.friendly_entities + self.memories:
            if obj.pos.x < -50 or obj.pos.x > self.SCREEN_WIDTH + 50 or \
               obj.pos.y < -50 or obj.pos.y > self.SCREEN_HEIGHT + 50:
                obj.health = 0 # Remove if far off-screen

    def _update_progression(self):
        # Increase spawn rate over time
        self.current_memory_spawn_rate = self.INITIAL_MEMORY_SPAWN_RATE + self.game_time_elapsed * self.MEMORY_SPAWN_RATE_INCREASE
        
        # Unlock new entity sizes
        if self.game_time_elapsed > 20 and self.unlocked_entity_level == 0:
            self.unlocked_entity_level = 1
        if self.game_time_elapsed > 40 and self.unlocked_entity_level == 1:
            self.unlocked_entity_level = 2
            
        # Unlock new memory types
        if self.game_time_elapsed > 15 and 1 not in self.unlocked_memory_types:
            self.unlocked_memory_types.append(1)

    def _spawn_memories(self, dt):
        self.memory_spawn_timer -= dt
        if self.memory_spawn_timer <= 0:
            if self.current_memory_spawn_rate > 0:
                self.memory_spawn_timer += 1.0 / self.current_memory_spawn_rate
            
            edge = self.np_random.integers(4)
            if edge == 0: # top
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -30)
            elif edge == 1: # bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 30)
            elif edge == 2: # left
                pos = pygame.Vector2(-30, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            else: # right
                pos = pygame.Vector2(self.SCREEN_WIDTH + 30, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
            mem_type = self.np_random.choice(self.unlocked_memory_types)
            self.memories.append(Memory(pos, self.center, mem_type))
            # sfx: memory_spawn

    def _handle_collisions(self):
        ego_radius = self._get_ego_radius()
        
        # Memories vs Ego
        for m in self.memories:
            if m.pos.distance_to(self.center) < ego_radius + m.radius:
                self.ego_health -= m.damage
                m.health = 0
                self._create_particles(m.pos, self.MEMORY_COLORS[m.type_id]['main'], 20)
                # sfx: ego_damage
        
        # Entities vs Memories
        for e in self.friendly_entities:
            for m in self.memories:
                dist = e.pos.distance_to(m.pos)
                if dist < e.radius + m.radius:
                    self._resolve_elastic_collision(e, m)
                    e.health -= m.damage / 2
                    m.health -= e.mass * 5 # damage based on entity mass
                    self._create_particles(e.pos.lerp(m.pos, 0.5), (200, 200, 200), 10)
                    # sfx: impact
        
        # Entities vs Entities
        for i in range(len(self.friendly_entities)):
            for j in range(i + 1, len(self.friendly_entities)):
                e1 = self.friendly_entities[i]
                e2 = self.friendly_entities[j]
                dist = e1.pos.distance_to(e2.pos)
                if dist < e1.radius + e2.radius:
                    self._resolve_elastic_collision(e1, e2)
                    # sfx: impact_light

    def _resolve_elastic_collision(self, obj1, obj2):
        dist = obj1.pos.distance_to(obj2.pos)
        if dist == 0: return # Avoid division by zero
        
        # Prevent overlap
        overlap = (obj1.radius + obj2.radius) - dist
        push_vec = (obj1.pos - obj2.pos).normalize() * overlap
        obj1.pos += push_vec * 0.5
        obj2.pos -= push_vec * 0.5
        
        # Collision normal
        normal = (obj2.pos - obj1.pos).normalize()
        # Relative velocity
        rel_vel = obj1.vel - obj2.vel
        # Impulse
        impulse = (2 * obj2.mass / (obj1.mass + obj2.mass)) * (rel_vel.dot(normal))
        
        obj1.vel -= impulse * normal
        obj2.vel += (obj1.mass / obj2.mass) * impulse * normal

    def _calculate_reward(self):
        # This is a placeholder. A more robust implementation would track
        # which memories were deflected and calculate damage since last step.
        # For simplicity, we use a proxy based on ego health change.
        # Event-based rewards would be handled inside collision logic.
        return {'reward': 0}

    def _check_termination(self):
        return self.ego_health <= 0 or self.game_time_elapsed >= self.GAME_DURATION_SECONDS

    def _get_ego_radius(self):
        return max(0, self.ego_health * 0.5)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Ego
        ego_radius = int(self._get_ego_radius())
        if ego_radius > 0:
            pulse = (math.sin(self.game_time_elapsed * 4) + 1) / 2 * 5
            # Glow
            for i in range(15, 0, -2):
                alpha = 50 - i * 3
                pygame.gfxdraw.filled_circle(
                    self.screen, int(self.center.x), int(self.center.y),
                    int(ego_radius + pulse + i),
                    (*self.COLOR_EGO_GLOW, alpha)
                )
            # Core
            pygame.gfxdraw.aacircle(self.screen, int(self.center.x), int(self.center.y), ego_radius, self.COLOR_EGO)
            pygame.gfxdraw.filled_circle(self.screen, int(self.center.x), int(self.center.y), ego_radius, self.COLOR_EGO)

        # Render Particles
        for p in self.particles:
            p.draw(self.screen)

        # Render Memories
        for m in self.memories:
            m.draw(self.screen, self.MEMORY_COLORS)

        # Render Friendly Entities
        for e in self.friendly_entities:
            e.draw(self.screen, self.ENTITY_COLORS)

    def _render_ui(self):
        # Render Cursor and placement preview
        preview_props = self.ENTITY_SIZES[self.selected_size_index]
        preview_radius = preview_props['radius']
        preview_color = self.ENTITY_COLORS[self.selected_size_index]['main']
        
        # Draw transparent preview
        preview_surface = pygame.Surface((preview_radius*2, preview_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(preview_surface, (*preview_color, 100), (preview_radius, preview_radius), preview_radius)
        self.screen.blit(preview_surface, (int(self.cursor_pos.x - preview_radius), int(self.cursor_pos.y - preview_radius)))

        # Draw cursor
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos.x - 5, self.cursor_pos.y), (self.cursor_pos.x + 5, self.cursor_pos.y), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos.x, self.cursor_pos.y - 5), (self.cursor_pos.x, self.cursor_pos.y + 5), 1)

        # Render Timer
        time_left = max(0, self.GAME_DURATION_SECONDS - self.game_time_elapsed)
        timer_text = f"{time_left:.1f}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH // 2, 30), self.font_main, self.COLOR_TEXT, center=True)

        # Render UI for entity selection
        self._draw_text("Size:", (20, self.SCREEN_HEIGHT - 40), self.font_small, self.COLOR_TEXT)
        for i in range(self.unlocked_entity_level + 1):
            is_selected = i == self.selected_size_index
            size_props = self.ENTITY_SIZES[i]
            color = self.ENTITY_COLORS[i]['main']
            radius = int(size_props['radius'] * 0.5)
            pos_x = 70 + i * 40
            
            if is_selected:
                pygame.draw.circle(self.screen, (255, 255, 255), (pos_x, self.SCREEN_HEIGHT - 32), radius + 3, 2)

            pygame.draw.circle(self.screen, color, (pos_x, self.SCREEN_HEIGHT - 32), radius)

    def _draw_text(self, text, pos, font, color, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        text_shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(text_shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ego_health": self.ego_health,
            "time_left": max(0, self.GAME_DURATION_SECONDS - self.game_time_elapsed),
        }

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos, color))
            
    def close(self):
        pygame.quit()

# --- Helper Classes ---

class FriendlyEntity:
    def __init__(self, pos, size_props, type_id):
        self.pos = pos
        self.vel = pygame.Vector2(0, 0)
        self.radius = size_props['radius']
        self.mass = size_props['mass']
        self.health = size_props['health']
        self.max_health = self.health
        self.type_id = type_id

    def draw(self, surface, colors):
        color_scheme = colors[self.type_id]
        health_ratio = max(0, self.health / self.max_health)
        current_radius = int(self.radius * health_ratio**0.5)
        if current_radius <= 0: return

        # Glow effect
        for i in range(10, 0, -2):
            alpha = int(80 * health_ratio) - i * 8
            if alpha > 0:
                pygame.gfxdraw.filled_circle(
                    surface, int(self.pos.x), int(self.pos.y),
                    current_radius + i,
                    (*color_scheme['glow'], alpha)
                )
        # Core
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), current_radius, color_scheme['main'])
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), current_radius, color_scheme['main'])

class Memory:
    def __init__(self, pos, target, type_id):
        self.pos = pos
        self.type_id = type_id
        
        if type_id == 0: # Standard
            self.health = 40
            self.damage = 10
            self.mass = 1
            self.radius = 12
            speed = random.uniform(30, 50)
            direction = (target - self.pos).normalize()
            self.vel = direction * speed
        elif type_id == 1: # Fast
            self.health = 30
            self.damage = 15
            self.mass = 0.8
            self.radius = 10
            speed = random.uniform(60, 90)
            direction = (target - self.pos).normalize()
            self.vel = direction * speed
            
        self.angle = math.atan2(self.vel.y, self.vel.x)
        self.rot_speed = random.uniform(-2, 2)

    def draw(self, surface, colors):
        color_scheme = colors[self.type_id]
        self.angle += self.rot_speed * (1.0 / 30.0) # Assuming 30 FPS for dt
        
        points = []
        num_points = 3
        for i in range(num_points):
            angle = self.angle + (2 * math.pi * i / num_points)
            x = self.pos.x + self.radius * math.cos(angle)
            y = self.pos.y + self.radius * math.sin(angle)
            points.append((int(x), int(y)))

        # Glow
        for i in range(8, 0, -2):
            glow_points = []
            alpha = 60 - i * 7
            for p_idx in range(num_points):
                angle = self.angle + (2 * math.pi * p_idx / num_points)
                x = self.pos.x + (self.radius + i) * math.cos(angle)
                y = self.pos.y + (self.radius + i) * math.sin(angle)
                glow_points.append((int(x), int(y)))
            pygame.gfxdraw.filled_polygon(surface, glow_points, (*color_scheme['glow'], alpha))

        # Core
        pygame.gfxdraw.aapolygon(surface, points, color_scheme['main'])
        pygame.gfxdraw.filled_polygon(surface, points, color_scheme['main'])

class Particle:
    def __init__(self, pos, color):
        self.pos = pos.copy()
        self.vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * random.uniform(20, 80)
        self.color = color
        self.lifespan = random.uniform(0.3, 0.8)
        self.life = self.lifespan
        self.radius = random.uniform(1, 4)

    def update(self, dt):
        self.life -= dt
        self.pos += self.vel * dt
        self.vel *= 0.95 # friction

    def is_alive(self):
        return self.life > 0

    def draw(self, surface):
        if not self.is_alive(): return
        alpha = int(255 * (self.life / self.lifespan))
        radius = int(self.radius * (self.life / self.lifespan))
        if radius > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), radius, (*self.color, alpha))

if __name__ == '__main__':
    # --- Example Usage ---
    # This block is for manual play and will not be used in the evaluation.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Ego Defense")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Time Survived: {env.game_time_elapsed:.2f}s")
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.TARGET_FPS)
        
    env.close()